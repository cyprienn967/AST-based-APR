import json
from collections.abc import Generator
from pathlib import Path

from loguru import logger
import re

from app.agents import agent_reviewer
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import TestAgent, TestHandle
from app.agents.agent_reviewer import Review, ReviewDecision
from app.agents.agent_write_ast import ASTAgent
from app.data_structures import MessageThread, ReproResult
from app.log import print_acr, print_review
from app.task import SweTask, Task
from app import config


# Type alias for patch handle
PatchHandle = str


def select_file_from_issue(issue_text: str, candidate_files: list[Path]) -> Path:
    """
    Select the most relevant file from candidates based on issue text.
    
    Uses keyword matching to find files/classes/functions mentioned in the issue.
    
    Args:
        issue_text: The issue description/statement
        candidate_files: List of Path objects for Python files
    
    Returns:
        The most relevant Path, or the first file if no matches found
    """
    if not candidate_files:
        raise ValueError("No candidate files provided")
    
    # Extract potential file/class/function names from issue
    # Look for patterns like: "in file.py", "class ClassName", "function func_name", etc.
    file_mentions = re.findall(r'(?:file|module|script)\s+[\w./]+\.py', issue_text, re.IGNORECASE)
    class_mentions = re.findall(r'(?:class|cls)\s+([A-Z][a-zA-Z0-9_]*)', issue_text)
    func_mentions = re.findall(r'(?:function|func|method|def)\s+([a-z_][a-z0-9_]*)', issue_text, re.IGNORECASE)
    
    # Also look for Python identifiers (PascalCase for classes, snake_case for functions/files)
    pascal_case = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', issue_text)
    snake_case = re.findall(r'\b([a-z_][a-z0-9_]{3,})\b', issue_text)
    
    all_keywords = set()
    all_keywords.update(class_mentions)
    all_keywords.update(func_mentions)
    all_keywords.update(pascal_case[:10])  # Limit to avoid noise
    all_keywords.update(snake_case[:20])
    
    # Remove common words
    common_words = {'the', 'this', 'that', 'with', 'from', 'have', 'should', 'would', 
                    'could', 'when', 'where', 'which', 'their', 'there', 'these', 'those',
                    'about', 'after', 'before', 'error', 'issue', 'problem', 'function',
                    'method', 'class', 'module', 'file', 'code', 'return', 'value'}
    all_keywords = {kw for kw in all_keywords if kw.lower() not in common_words}
    
    # Score each file by keyword matches
    file_scores = {}
    for file_path in candidate_files:
        score = 0
        file_content = file_path.stem  # filename without extension
        
        try:
            # Read file to check for class/function definitions
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(50000)  # First 50KB
                
                # Check for keyword matches in content
                for keyword in all_keywords:
                    # Filename match (highest weight)
                    if keyword.lower() in file_content.lower():
                        score += 10
                    
                    # Class definition match
                    if re.search(rf'\bclass\s+{re.escape(keyword)}\b', content):
                        score += 5
                    
                    # Function definition match
                    if re.search(rf'\bdef\s+{re.escape(keyword)}\b', content):
                        score += 3
                    
                    # General mention in code
                    if keyword in content:
                        score += 1
        except Exception:
            # If we can't read the file, just use filename matching
            for keyword in all_keywords:
                if keyword.lower() in file_content.lower():
                    score += 5
        
        file_scores[file_path] = score
    
    # Return file with highest score
    if file_scores:
        best_file = max(file_scores.items(), key=lambda x: x[1])
        if best_file[1] > 0:
            logger.info(f"Selected file {best_file[0]} based on issue analysis (score: {best_file[1]})")
            return best_file[0]
    
    # Default to first file
    logger.warning(f"No keyword matches; using first file: {candidate_files[0]}")
    return candidate_files[0]


class ReviewManager:
    def __init__(
        self,
        task: Task,
        output_dir: str,
        test_agent: TestAgent,
        repro_result_map: (
            dict[tuple[PatchHandle, TestHandle], ReproResult] | None
        ) = None,
        sbfl_line_scores: dict[str, dict[int, float]] | None = None,
        traceback_text: str = "",
    ) -> None:
        self.issue_stmt = task.get_issue_statement()
        self.ast_agent = ASTAgent(
            task,
            self.issue_stmt,
            output_dir,
        )
        self.test_agent = test_agent
        self.task: Task = task
        self.repro_result_map: dict[tuple[PatchHandle, TestHandle], ReproResult] = dict(
            repro_result_map or {}
        )
        self.output_dir = output_dir
        self._patch_handles: list[PatchHandle] = []
        
        # Store SBFL and traceback info for passing to ASTAgent
        self.sbfl_line_scores = sbfl_line_scores or {}
        self.traceback_text = traceback_text

    def patch_only_generator(
        self,
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        try:
            while True:
                # Get the main file to patch
                file_path = self._get_buggy_file()
                if not file_path:
                    raise InvalidLLMResponse("Could not determine file to patch")
                
                # Get SBFL scores for this specific file
                file_sbfl_scores = self.sbfl_line_scores.get(file_path, {})
                
                success, response, patch_content = self.ast_agent.write_patch_for_file(
                    file_path,
                    sbfl_line_scores=file_sbfl_scores,
                    traceback_text=self.traceback_text,
                    enable_fast_path=config.enable_micro_edit_fast_path,
                )
                
                if not success:
                    raise InvalidLLMResponse(f"Patch generation failed: {response}")
                
                patch_handle = f"ast_patch_{len(self._patch_handles)}"
                self._patch_handles.append(patch_handle)
                self.save_patch(patch_handle, patch_content)

                yield patch_handle, patch_content
        except InvalidLLMResponse as e:
            logger.info("Aborting patch-only with exception: {}", str(e))

    def generator(
        self, rounds: int = 5
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        """
        This is the generator when reproducer is available.
        """
        assert isinstance(
            self.task, SweTask
        ), "Only SweTask is supported for reproducer+patch generator."

        try:
            yield from self._generator(rounds)
        except InvalidLLMResponse as e:
            logger.info("Aborting review with exception: {}", str(e))

    def _generator(
        self, rounds: int
    ) -> Generator[tuple[PatchHandle, str], str | None, None]:
        issue_statement = self.task.get_issue_statement()

        # TODO: fall back to iterative patch generation when reproduction fails
        if not self.test_agent._history:
            (
                test_handle,
                test_content,
                orig_repro_result,
            ) = self.test_agent.write_reproducing_test_without_feedback()
            self.test_agent.save_test(test_handle)
        else:
            test_handle = self.test_agent._history[-1]
            test_content = self.test_agent._tests[test_handle]
            orig_repro_result = self.repro_result_map[
                ("EMPTY_PATCH", test_handle)
            ]

        coords = ("EMPTY_PATCH", test_handle)
        self.repro_result_map[coords] = orig_repro_result
        self.save_execution_result(orig_repro_result, *coords)

        # write the first patch
        file_path = self._get_buggy_file()
        if not file_path:
            raise InvalidLLMResponse("Could not determine file to patch")
        
        # Get SBFL scores for this specific file
        file_sbfl_scores = self.sbfl_line_scores.get(file_path, {})
        
        success, response, patch_content = self.ast_agent.write_patch_for_file(
            file_path,
            sbfl_line_scores=file_sbfl_scores,
            traceback_text=self.traceback_text,
        )
        
        if not success:
            raise InvalidLLMResponse(f"Patch generation failed: {response}")
        
        patch_handle = f"ast_patch_{len(self._patch_handles)}"
        self._patch_handles.append(patch_handle)
        self.save_patch(patch_handle, patch_content)

        for _ in range(rounds):
            patched_repro_result = self.task.execute_reproducer(
                test_content, patch_content
            )

            coords = (patch_handle, test_handle)
            self.repro_result_map[coords] = patched_repro_result
            self.save_execution_result(patched_repro_result, *coords)

            review, review_thread = agent_reviewer.run(
                issue_statement,
                test_content,
                patch_content,
                orig_repro_result,
                patched_repro_result,
            )

            print_review(str(review))
            self.save_review(patch_handle, test_handle, review)
            review_thread.save_to_file(
                Path(self.output_dir, f"conv_review_{patch_handle}_{test_handle}.json")
            )

            if review.patch_decision == ReviewDecision.YES:
                evaluation_msg = yield patch_handle, patch_content
                assert evaluation_msg is not None

                print_acr(evaluation_msg, "Patch evaluation")

            if review.patch_decision == ReviewDecision.NO:
                # Generate new patch (feedback is not used in AST-based approach)
                file_path = self._get_buggy_file()
                if not file_path:
                    raise InvalidLLMResponse("Could not determine file to patch")
                
                # Get SBFL scores for this specific file
                file_sbfl_scores = self.sbfl_line_scores.get(file_path, {})
                
                success, response, patch_content = self.ast_agent.write_patch_for_file(
                    file_path,
                    sbfl_line_scores=file_sbfl_scores,
                    traceback_text=self.traceback_text,
                    enable_fast_path=config.enable_micro_edit_fast_path,
                )
                
                if not success:
                    raise InvalidLLMResponse(f"Patch generation failed: {response}")
                
                patch_handle = f"ast_patch_{len(self._patch_handles)}"
                self._patch_handles.append(patch_handle)
                self.save_patch(patch_handle, patch_content)

            if review.test_decision == ReviewDecision.NO:
                feedback = self.compose_feedback_for_test_generation(
                    review, patch_content
                )
                self.test_agent.add_feedback(test_handle, feedback)
                (
                    test_handle,
                    test_content,
                    orig_repro_result,
                ) = self.test_agent.write_reproducing_test_with_feedback()
                self.test_agent.save_test(test_handle)
                coords = ("EMPTY_PATCH", test_handle)
                self.repro_result_map[coords] = orig_repro_result
                self.save_execution_result(orig_repro_result, *coords)

    def _get_buggy_file(self) -> str | None:
        """
        Determine the file to patch from the task.
        Priority:
        1. Explicit file_to_edit attribute
        2. File with highest SBFL score (using improved scoring)
        3. File matching issue statement keywords (semantic selection)
        4. First Python file in repo
        """
        if hasattr(self.task, 'file_to_edit'):
            return self.task.file_to_edit
        
        # Use SBFL results to find the most suspicious file
        if self.sbfl_line_scores:
            file_scores = self._compute_improved_sbfl_scores()
            
            if file_scores:
                most_suspicious_file = max(file_scores.items(), key=lambda x: x[1])[0]
                logger.info("Selected file {} based on improved SBFL (score: {:.2f})", 
                          most_suspicious_file, file_scores[most_suspicious_file])
                return most_suspicious_file
        
        # Fallback: use issue statement to find relevant files
        if hasattr(self.task, 'project_path'):
            project_path = Path(self.task.project_path)
            all_py_files = [
                f for f in project_path.glob("**/*.py") 
                if "__pycache__" not in str(f) and "test" not in str(f).lower()
            ]
            
            if all_py_files:
                try:
                    # Use issue analysis to select most relevant file
                    selected_file = select_file_from_issue(
                        self.issue_stmt,
                        all_py_files
                    )
                    return str(selected_file)
                except Exception as e:
                    logger.warning("Issue-based file selection failed: {}; using first file", e)
                    return str(all_py_files[0])
        
        return None
    
    def _compute_improved_sbfl_scores(self) -> dict[str, float]:
        """
        Compute improved SBFL scores that avoid biasing toward large/framework files.
        
        Strategy:
        1. Filter out framework/test files (conftest, __init__, setup, etc.)
        2. Use max line score + normalized average (not sum) to avoid size bias
        3. Penalize very large files unless they have extremely high max scores
        4. Boost files that match keywords from issue statement
        
        Returns:
            dict mapping file_path -> composite score
        """
        file_scores = {}
        
        # Extract keywords from issue statement for boosting
        issue_keywords = self._extract_issue_keywords()
        
        for file_path, line_scores in self.sbfl_line_scores.items():
            # Filter out framework and test files
            if self._is_framework_file(file_path):
                logger.debug("Skipping framework file: {}", file_path)
                continue
            
            if not line_scores:
                continue
            
            # Get file size (number of lines)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    num_lines = sum(1 for _ in f)
            except Exception:
                num_lines = len(line_scores)
            
            # Penalize very large files (>1000 lines) - they're usually core modules
            if num_lines > 1000:
                size_penalty = 0.5  # Reduce score by 50%
                logger.debug("Applying size penalty to large file: {} ({} lines)", file_path, num_lines)
            else:
                size_penalty = 1.0
            
            # Compute metrics
            max_score = max(line_scores.values())
            avg_score = sum(line_scores.values()) / len(line_scores)
            num_suspicious_lines = sum(1 for s in line_scores.values() if s > 0.7)
            
            # Composite score: prioritize max score (indicates critical line),
            # but also consider density of suspicious lines
            # Formula: max_score * 2 + avg_score + (num_suspicious_lines / 100)
            base_score = (max_score * 2.0) + avg_score + (num_suspicious_lines / 100.0)
            
            # Apply size penalty
            base_score *= size_penalty
            
            # Boost if file matches issue keywords
            issue_boost = self._compute_issue_match_boost(file_path, issue_keywords)
            final_score = base_score * (1.0 + issue_boost)
            
            file_scores[file_path] = final_score
            logger.debug("File: {} | max={:.2f} avg={:.2f} suspicious_lines={} size_penalty={:.1f} issue_boost={:.2f} final={:.2f}",
                        file_path, max_score, avg_score, num_suspicious_lines, size_penalty, issue_boost, final_score)
        
        return file_scores
    
    def _is_framework_file(self, file_path: str) -> bool:
        """Check if a file is likely a framework/infrastructure file rather than application code."""
        file_path_lower = file_path.lower()
        filename = Path(file_path).name.lower()
        
        # Exclude common framework patterns
        framework_patterns = [
            'conftest.py',
            '__init__.py',
            'setup.py',
            'test_',
            '_test.py',
            '/tests/',
            '/testing/',
            'fixtures.py',
            'config.py',
            'settings.py',
        ]
        
        for pattern in framework_patterns:
            if pattern in file_path_lower or pattern in filename:
                return True
        
        return False
    
    def _extract_issue_keywords(self) -> set[str]:
        """Extract relevant keywords from the issue statement."""
        if not hasattr(self, 'issue_stmt') or not self.issue_stmt:
            return set()
        
        # Extract PascalCase identifiers (likely class names)
        pascal_case = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', self.issue_stmt)
        
        # Extract snake_case identifiers (likely function/module names)
        snake_case = re.findall(r'\b([a-z_][a-z0-9_]{3,})\b', self.issue_stmt)
        
        # Extract words from quotes (often specific identifiers)
        quoted = re.findall(r'[\'"`]([a-zA-Z_][a-zA-Z0-9_]+)[\'"`]', self.issue_stmt)
        
        keywords = set(pascal_case[:10] + snake_case[:15] + quoted[:10])
        
        # Remove common English words
        common_words = {'the', 'this', 'that', 'with', 'from', 'have', 'should', 
                       'would', 'could', 'when', 'where', 'which', 'their', 'there',
                       'about', 'error', 'issue', 'problem', 'function', 'method',
                       'class', 'module', 'file', 'code', 'return', 'value', 'need',
                       'want', 'expect', 'expected', 'actual', 'result', 'output'}
        keywords = {kw for kw in keywords if kw.lower() not in common_words}
        
        return keywords
    
    def _compute_issue_match_boost(self, file_path: str, keywords: set[str]) -> float:
        """
        Compute a boost factor based on how well the file matches issue keywords.
        Returns a value between 0.0 and 1.0 (where 1.0 means double the score).
        """
        if not keywords:
            return 0.0
        
        filename = Path(file_path).stem.lower()
        boost = 0.0
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            
            # Strong match: keyword in filename
            if kw_lower in filename or filename in kw_lower:
                boost += 0.5
            
            # Medium match: keyword in file path
            elif kw_lower in file_path.lower():
                boost += 0.2
        
        # Cap the boost at 1.0 (would double the score)
        return min(boost, 1.0)

    @classmethod
    def compose_feedback_for_test_generation(cls, review: Review, patch: str) -> str:
        return (
            f"Here is a patch to the program:\n"
            "```\n"
            f"{patch}"
            "```\n"
            f"By executing your test with and without the patch,"
            " the following analysis can be made:\n"
            "\n"
            f"{review.test_analysis}"
            "\n"
            "Therefore, the test does not correctly reproduce the issue.\n"
            "\n"
            "To correct the test, here is my advice:\n"
            "\n"
            f"{review.test_advice}"
        )

    def save_patch(self, handle: PatchHandle, content: str) -> None:
        Path(self.output_dir, f"extracted_patch_{handle}.diff").write_text(content)

    def save_test(self, handle: TestHandle, content: str) -> None:
        Path(self.output_dir, f"reproducer_{handle}.py").write_text(content)

    def save_review(
        self, patch_handle: PatchHandle, test_handle: TestHandle, review: Review
    ) -> None:
        path = Path(self.output_dir, f"review_p{patch_handle}_t{test_handle}.json")
        path.write_text(json.dumps(review.to_json(), indent=4))

    def save_execution_result(
        self, result: ReproResult, patch_handle: str, test_handle: str
    ) -> None:
        Path(
            self.output_dir, f"execution_{patch_handle}_{test_handle}.json"
        ).write_text(
            json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "triggered": result.reproduced,
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    pass
