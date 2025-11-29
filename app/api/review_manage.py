import json
from collections.abc import Generator
from pathlib import Path

from loguru import logger

from app.agents import agent_reviewer
from app.agents.agent_common import InvalidLLMResponse
from app.agents.agent_reproducer import TestAgent, TestHandle
from app.agents.agent_reviewer import Review, ReviewDecision
from app.agents.agent_write_ast import ASTAgent
from app.data_structures import MessageThread, ReproResult
from app.log import print_acr, print_review
from app.task import SweTask, Task


# Type alias for patch handle
PatchHandle = str


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
        2. File with highest SBFL score
        3. First Python file in repo
        """
        if hasattr(self.task, 'file_to_edit'):
            return self.task.file_to_edit
        
        # Use SBFL results to find the most suspicious file
        if self.sbfl_line_scores:
            # Find file with highest total suspiciousness
            file_scores = {}
            for file_path, line_scores in self.sbfl_line_scores.items():
                total_score = sum(line_scores.values())
                file_scores[file_path] = total_score
            
            if file_scores:
                most_suspicious_file = max(file_scores.items(), key=lambda x: x[1])[0]
                logger.info("Selected file {} based on SBFL (score: {})", 
                          most_suspicious_file, file_scores[most_suspicious_file])
                return most_suspicious_file
        
        # Fallback: find any Python file
        if hasattr(self.task, 'project_path'):
            project_path = Path(self.task.project_path)
            for py_file in project_path.glob("**/*.py"):
                if "__pycache__" not in str(py_file):
                    logger.warning("No SBFL data; using first Python file: {}", py_file)
                    return str(py_file)
        
        return None

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
