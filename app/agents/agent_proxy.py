"""
A proxy agent that extracts API calls and bug locations from text into JSON format.

This version is hardened:
- Uses SELECTED_MODEL.call correctly (response_format="json_object")
- Robust JSON recovery from messy LLM output
- Better logging and validation
- Fully backward compatible with existing pipelines
"""

import inspect
from typing import Any

from loguru import logger

from app.data_structures import MessageThread
from app.model import common
from app.post_process import ExtractStatus, is_valid_json
from app.search.search_backend import SearchBackend
from app.utils import parse_function_invocation


PROXY_PROMPT = """
You are a helpful assistant that extracts API calls and bug locations from text and outputs JSON only.

Input consists of two parts:
1. Whether more context is needed.
2. Bug locations.

Extract:
- API calls from section 1 (leave empty list if none)
- Bug locations from section 2 (leave empty list if none)

Valid API calls (must be emitted as valid python expressions, no placeholder arguments):
    search_method_in_class(method_name: str, class_name: str)
    search_method_in_file(method_name: str, file_path: str)
    search_method(method_name: str)
    search_class_in_file(class_name: str, file_path: str)
    search_class(class_name: str)
    search_code_in_file(code_str: str, file_path: str)
    search_code(code_str: str)
    get_code_around_line(file_path: str, line_number: int, window_size: int)

Output JSON structure:

{
  "API_calls": ["api_call_1(...)", "api_call_2(...)", ...],
  "bug_locations": [
      {
        "file": "path/to/file",
        "class": "ClassName",
        "method": "method_name",
        "intended_behavior": "Describe exactly how this code should behave after being fixed."
      },
      ...
  ]
}

Each bug location MUST include a non-empty intended_behavior field.
If you do not know the file/class/method, use an empty string for that field.
Return ONLY JSON. No explanation.
"""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_region(raw: str) -> str:
    """
    Extract JSON region ([[{...}]]) from noisy LLM output.
    """
    raw = raw.strip()
    first_brace = raw.find("{")
    first_bracket = raw.find("[")

    candidates = [i for i in [first_brace, first_bracket] if i != -1]
    if not candidates:
        raise ValueError("No JSON start token found in model output")

    start = min(candidates)

    last_brace = raw.rfind("}")
    last_bracket = raw.rfind("]")

    end = max(last_brace, last_bracket)
    if end == -1:
        raise ValueError("No JSON end token found in model output")

    return raw[start:end + 1]


# ---------------------------------------------------------------------------
# Main agent APIs
# ---------------------------------------------------------------------------

def run_with_retries(text: str, retries=5) -> tuple[str | None, list[MessageThread]]:
    """
    Try multiple times to get valid JSON output from the proxy agent.
    Returns (json_string or None, [MessageThreads]).
    """
    msg_threads = []

    for idx in range(1, retries + 1):
        logger.debug(f"Proxy agent: attempt {idx}/{retries}")

        res_text, msg_thread = run(text)
        msg_threads.append(msg_thread)

        status, data = is_valid_json(res_text)
        if status != ExtractStatus.IS_VALID_JSON:
            logger.debug("LLM output not valid JSON: retrying")
            continue

        valid, diag = is_valid_response(data)
        if not valid:
            logger.debug(f"Invalid proxy response ({diag}): retrying")
            continue

        logger.debug("Proxy agent extracted valid JSON")
        return res_text, msg_threads

    logger.debug("Proxy agent failed after retries.")
    return None, msg_threads


def run(text: str) -> tuple[str, MessageThread]:
    """
    Send user text to LLM and extract JSON describing
    API calls and bug locations.
    """

    msg_thread = MessageThread()
    msg_thread.add_system(PROXY_PROMPT)
    msg_thread.add_user(text)

    # NOTE: Use response_format="json_object" if supported.
    # LiteLLM will send OpenAI-style {"type":"json_object"} for GPTs,
    # and None for non-GPT models (completely safe).
    try:
        if common.SELECTED_MODEL is None:
            raise RuntimeError("SELECTED_MODEL is not initialized")
        model_resp, *_ = common.SELECTED_MODEL.call(
            msg_thread.to_msg(),
            response_format="json_object",
        )
    except Exception as exc:
        logger.error(f"Proxy agent model error: {exc}")
        return "{}", msg_thread

    # Recover JSON region from possibly noisy output
    try:
        cleaned = _extract_json_region(model_resp)
    except Exception:
        cleaned = model_resp  # worst case: attempt raw

    msg_thread.add_model(cleaned, tools=[])

    return cleaned, msg_thread


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def is_valid_response(data: Any) -> tuple[bool, str]:
    """
    Schema validation for proxy output.
    """

    if not isinstance(data, dict):
        return False, "Json is not a dict"

    # Validate bug_locations or API_calls exist
    api_calls = data.get("API_calls")
    bug_locations = data.get("bug_locations")

    if not api_calls:
        # Must have bug locations then
        if not isinstance(bug_locations, list) or not bug_locations:
            return False, "Both API_calls and bug_locations are empty"

        # Validate bug locations minimally
        for loc in bug_locations:
            if not isinstance(loc, dict):
                return False, "Bug location is not a dict"

            if not (loc.get("file") or loc.get("class") or loc.get("method")):
                return False, "Bug location not detailed enough (need file/class/method)"

            intended = loc.get("intended_behavior", "")
            if not isinstance(intended, str) or not intended.strip():
                return False, "Each bug location must include intended_behavior text"

        return True, "OK"

    # Validate API calls
    if not isinstance(api_calls, list):
        return False, "API_calls must be a list"

    for api_call in api_calls:
        if not isinstance(api_call, str):
            return False, "Every API call must be a string"

        try:
            func_name, func_args = parse_function_invocation(api_call)
        except Exception:
            return False, "Every API call must be of form func_name(arg1, ...)"

        backend_fn = getattr(SearchBackend, func_name, None)
        if backend_fn is None:
            return False, f"API call {func_name} calls a non-existent function"

        # unwrap decorators
        while "__wrapped__" in getattr(backend_fn, "__dict__", {}):
            backend_fn = backend_fn.__wrapped__

        arg_spec = inspect.getfullargspec(backend_fn)
        expected = arg_spec.args[1:]  # skip self

        if len(func_args) != len(expected):
            return False, (
                f"API call '{api_call}' has wrong number of arguments. "
                f"Expected {len(expected)}, got {len(func_args)}."
            )

    return True, "OK"
