import ast
import json
from pathlib import Path
from typing import Sequence

from loguru import logger

from app.ast_repair.localize import localize_fault
from app.ast_repair.localization import stacktrace_anchor
from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.parser import parse_file_to_ast
from app.code_utils import read_code_snippet
from app.data_structures import BugLocation, MessageThread, SearchResult


class SearchManager:
    """
    Legacy name retained for compatibility. This class now runs the AST-based
    localization orchestrator (app.ast_repair.localize.localize_fault) instead
    of invoking LLM search APIs.
    """

    MAX_FILES = 3
    MAX_NODES_PER_FILE = 3

    def __init__(self, project_path: str, output_dir: str):
        self.project_path = Path(project_path).resolve()
        self.project_root_str = str(self.project_path)
        self.output_dir = Path(output_dir, "search")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_call_layers: list[list[dict]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_iterative(
        self,
        task,
        sbfl_result: str,
        reproducer_result: str,
        reproduced_test_content: str | None,
        sbfl_ranked_lines: Sequence[tuple[str, int, float]] | None = None,
    ) -> tuple[list[BugLocation], MessageThread]:
        """
        Run AST localization across a handful of candidate files and
        convert the results to legacy BugLocation records.
        """

        ranked_lines = list(sbfl_ranked_lines or [])
        bug_locations: list[BugLocation] = []

        self.tool_call_layers = []
        self.start_new_tool_call_layer()

        candidate_files = self._collect_candidate_files(
            ranked_lines, reproducer_result
        )
        issue_stmt = task.get_issue_statement().strip()

        for file_path in candidate_files:
            localized = self._localize_file(
                file_path,
                ranked_lines,
                reproducer_result or "",
                issue_stmt,
            )
            bug_locations.extend(localized)

            try:
                rel_path = file_path.relative_to(self.project_path)
            except ValueError:
                rel_path = file_path
            self.add_tool_call_to_curr_layer(
                "localize_fault",
                {
                    "file": str(rel_path),
                    "sbfl_hits": len(self._sbfl_scores_for_file(file_path, ranked_lines)),
                },
                bool(localized),
            )

        thread = self._build_thread_summary(bug_locations, sbfl_result)

        return bug_locations, thread

    def start_new_tool_call_layer(self):
        self.tool_call_layers.append([])

    def add_tool_call_to_curr_layer(
        self, func_name: str, args: dict[str, str | int], result: bool
    ):
        if not self.tool_call_layers:
            self.tool_call_layers.append([])
        self.tool_call_layers[-1].append(
            {
                "func_name": func_name,
                "arguments": args,
                "call_ok": result,
            }
        )

    def dump_tool_call_layers_to_file(self):
        tool_call_file = Path(self.output_dir, "tool_call_layers.json")
        tool_call_file.write_text(json.dumps(self.tool_call_layers, indent=4))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_candidate_files(
        self,
        ranked_lines: Sequence[tuple[str, int, float]],
        traceback_text: str,
    ) -> list[Path]:
        """
        Pick a small set of files to run localization on:
            1. Top files from SBFL scores
            2. Fall back to traceback file if SBFL is empty
        """

        selected: list[Path] = []
        seen: set[Path] = set()

        for file_path, _line, _score in ranked_lines:
            normalized = self._normalize_path(file_path)
            if normalized is None or normalized in seen:
                continue
            if not normalized.exists():
                continue
            selected.append(normalized)
            seen.add(normalized)
            if len(selected) >= self.MAX_FILES:
                break

        if len(selected) < self.MAX_FILES:
            trace_candidate = self._candidate_from_traceback(traceback_text)
            if trace_candidate and trace_candidate not in seen:
                selected.append(trace_candidate)

        return selected

    def _candidate_from_traceback(self, traceback_text: str) -> Path | None:
        if not traceback_text:
            return None
        frame = stacktrace_anchor.extract_deepest_relevant_frame(
            traceback_text, self.project_root_str
        )
        if frame is None:
            return None
        file_path, _ = frame
        normalized = self._normalize_path(file_path)
        if normalized is None or not normalized.exists():
            return None
        return normalized

    def _normalize_path(self, raw_path: str | Path) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_path / path
        return path.resolve()

    def _localize_file(
        self,
        file_path: Path,
        ranked_lines: Sequence[tuple[str, int, float]],
        traceback_text: str,
        issue_stmt: str,
    ) -> list[BugLocation]:
        try:
            root, metadata = parse_file_to_ast(str(file_path))
        except Exception as exc:
            logger.warning("Failed to parse {}: {}", file_path, exc)
            return []

        sbfl_scores = self._sbfl_scores_for_file(file_path, ranked_lines)
        failing_line = self._pick_failing_line(file_path, sbfl_scores, traceback_text)

        bug_nodes = localize_fault(
            root=root,
            md=metadata,
            sbfl_line_scores=sbfl_scores,
            traceback_text=traceback_text or "",
            failing_line=failing_line,
            project_root=self.project_root_str,
            top_k=self.MAX_NODES_PER_FILE,
        )

        if not bug_nodes:
            return []

        bug_locations: list[BugLocation] = []
        for ast_loc in bug_nodes:
            bug_loc = self._build_bug_location(file_path, metadata, ast_loc, issue_stmt)
            if bug_loc is not None:
                bug_locations.append(bug_loc)

        return bug_locations

    def _sbfl_scores_for_file(
        self,
        file_path: Path,
        ranked_lines: Sequence[tuple[str, int, float]],
    ) -> dict[int, float]:
        target = str(file_path)
        scores: dict[int, float] = {}
        for raw_path, line, score in ranked_lines:
            normalized = self._normalize_path(raw_path)
            if normalized is None:
                continue
            if str(normalized) != target:
                continue
            scores[line] = max(scores.get(line, 0.0), score)
        return scores

    def _pick_failing_line(
        self,
        file_path: Path,
        sbfl_scores: dict[int, float],
        traceback_text: str,
    ) -> int | None:
        frame = stacktrace_anchor.extract_deepest_relevant_frame(
            traceback_text, self.project_root_str
        )
        if frame is not None:
            frame_path, line = frame
            normalized = self._normalize_path(frame_path)
            if normalized is not None and normalized == file_path:
                return line

        if not sbfl_scores:
            return None

        return max(sbfl_scores.items(), key=lambda item: item[1])[0]

    def _build_bug_location(
        self,
        file_path: Path,
        metadata: ASTMetadata,
        ast_loc,
        issue_stmt: str,
    ) -> BugLocation | None:
        start, end = metadata.get_line_span(ast_loc.node_id)
        if start is None or end is None:
            return None

        snippet = read_code_snippet(str(file_path), start, end, with_lineno=True)
        if not snippet.strip():
            return None

        class_name, method_name = self._resolve_symbols(metadata, ast_loc.node_id)
        search_res = SearchResult(
            str(file_path),
            start,
            end,
            class_name,
            method_name,
            snippet,
        )

        intended_behavior = self._format_intended_behavior(issue_stmt, class_name, method_name)

        try:
            return BugLocation(search_res, self.project_root_str, intended_behavior)
        except Exception as exc:
            logger.warning("Failed to build BugLocation for {}: {}", file_path, exc)
            return None

    def _resolve_symbols(
        self,
        metadata: ASTMetadata,
        node_id: int,
    ) -> tuple[str | None, str | None]:
        class_name = None
        method_name = None
        current = node_id

        while current is not None:
            node = metadata.get_node_by_id(current)
            if method_name is None and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                method_name = node.name
            if class_name is None and isinstance(node, ast.ClassDef):
                class_name = node.name
            current = metadata.get_parent(current)

            if class_name and method_name:
                break

        return class_name, method_name

    def _format_intended_behavior(
        self,
        issue_stmt: str,
        class_name: str | None,
        method_name: str | None,
    ) -> str:
        prefix = "This code should resolve the reported issue."
        symbols = []
        if class_name:
            symbols.append(f"class {class_name}")
        if method_name:
            symbols.append(f"method {method_name}")
        if symbols:
            prefix = f"The {'/'.join(symbols)} should resolve the reported issue."
        if issue_stmt:
            return f"{prefix}\n\nIssue context:\n{issue_stmt}"
        return prefix

    def _build_thread_summary(
        self,
        bug_locations: Sequence[BugLocation],
        sbfl_result: str,
    ) -> MessageThread:
        """
        Build a minimal MessageThread for compatibility. The AST patch agent
        prefers structured bug locations, so this thread rarely gets used.
        """
        thread = MessageThread()
        summary = (
            "AST localization has been applied automatically. "
            f"Identified {len(bug_locations)} suspicious regions."
        )
        thread.add_system(summary)
        if sbfl_result:
            thread.add_user(sbfl_result)
        return thread
