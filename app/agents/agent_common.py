"""
Common stuff for task agents.
"""

import re

from app.data_structures import MessageThread


def replace_system_prompt(msg_thread: MessageThread, prompt: str) -> MessageThread:
    """
    Replace the system prompt in the message thread.
    This is because the main agent system prompt main invole tool_calls info, which
    should not be known to task agents.
    """
    msg_thread.messages[0]["content"] = prompt
    return msg_thread


def extract_markdown_code_blocks(content: str) -> list[str]:
    """
    Extract code blocks from markdown-formatted content.
    
    Args:
        content: String potentially containing markdown code blocks
        
    Returns:
        List of code block contents (without the ``` markers)
    """
    lines = content.splitlines(keepends=True)

    in_code_block = False
    start_pattern = r"\s*```\w*\s*"
    end_pattern = r"\s*```\s*"

    start, end = -1, -1
    intervals = []

    for idx, line in enumerate(lines):
        if (not in_code_block) and re.match(start_pattern, line):
            in_code_block = True
            start = idx + 1
        elif in_code_block and re.match(end_pattern, line):
            in_code_block = False
            end = idx
            intervals.append((start, end))

    res = ["".join(lines[start:end]) for start, end in intervals]
    return res


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from an LLM response that might be wrapped in markdown code blocks.
    
    Args:
        response: LLM response string
        
    Returns:
        JSON string (either extracted from code block or original response)
    """
    code_blocks = extract_markdown_code_blocks(response)
    return code_blocks[0] if code_blocks else response


class InvalidLLMResponse(RuntimeError):
    pass
