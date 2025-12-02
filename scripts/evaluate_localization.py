#!/usr/bin/env python3
"""
Evaluate localization accuracy WITHOUT running expensive LLM calls.

This script:
1. Loads tasks from config file
2. Runs SBFL to get line-level suspiciousness scores
3. Runs file-level and node-level localization
4. Compares against ground truth from developer patches
5. Outputs comprehensive metrics

Usage:
    python scripts/evaluate_localization.py conf/vanilla-lite.conf

Metrics computed:
- File-level accuracy: Did we select the correct file?
- Method-level recall@K: What % of buggy methods are in top-K?
- Line-level recall@K: What % of buggy lines are within K lines of predictions?
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app import config
from app.analysis import sbfl
from app.raw_tasks import RawSweTask
from app.task import SweTask


@dataclass
class GroundTruth:
    """Parsed ground truth from developer patch."""
    file_path: str
    lines: Set[int] = field(default_factory=set)
    methods: Set[str] = field(default_factory=set)
    
    
@dataclass
class LocalizationResult:
    """Result of localization for a single task."""
    task_id: str
    
    # Ground truth
    gt_files: List[str] = field(default_factory=list)
    gt_lines: Dict[str, Set[int]] = field(default_factory=dict)
    gt_methods: Set[str] = field(default_factory=set)
    
    # Predictions
    predicted_file: Optional[str] = None
    predicted_lines: Dict[str, Set[int]] = field(default_factory=dict)
    sbfl_files: List[str] = field(default_factory=list)  # All files with SBFL scores
    
    # Metrics
    file_correct: bool = False
    line_recall: float = 0.0
    method_recall: float = 0.0
    
    # Error info
    error: Optional[str] = None


def parse_config(config_file: str) -> dict:
    """Parse the configuration file."""
    conf = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                conf[key.strip()] = value.strip().strip('"\'')
    return conf


def load_tasks(conf: dict) -> List[RawSweTask]:
    """Load tasks from configuration."""
    tasks_file = conf.get('selected_tasks_file')
    swe_bench_dir = conf.get('swe_bench_dir', '/opt/SWE-bench/')
    setup_result_dir = conf.get('setup_result_dir', os.path.join(swe_bench_dir, 'setup_result'))
    
    if not tasks_file:
        raise ValueError("No selected_tasks_file in config")
    
    # Read task IDs
    with open(tasks_file) as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    tasks = []
    
    # Load task metadata from setup_result
    for task_id in task_ids:
        setup_info_file = os.path.join(setup_result_dir, 'setup_map.json')
        task_info_file = os.path.join(setup_result_dir, 'tasks_map.json')
        
        if not os.path.exists(setup_info_file) or not os.path.exists(task_info_file):
            logger.warning(f"Missing setup files for {task_id}, skipping")
            continue
            
        with open(setup_info_file) as f:
            setup_map = json.load(f)
        with open(task_info_file) as f:
            tasks_map = json.load(f)
            
        if task_id not in setup_map or task_id not in tasks_map:
            logger.warning(f"Task {task_id} not in setup/tasks map, skipping")
            continue
            
        tasks.append(RawSweTask(task_id, setup_map[task_id], tasks_map[task_id]))
    
    return tasks


def parse_ground_truth_patch(patch_str: str) -> List[GroundTruth]:
    """
    Parse ground truth from developer patch.
    
    Returns list of GroundTruth objects, one per file modified.
    """
    results = []
    current_file = None
    current_gt = None
    
    for line in patch_str.split('\n'):
        # Match file header: "diff --git a/path/to/file.py b/path/to/file.py"
        if line.startswith('diff --git'):
            match = re.search(r'b/(.+?)(?:\s|$)', line)
            if match:
                if current_gt:
                    results.append(current_gt)
                current_file = match.group(1)
                current_gt = GroundTruth(file_path=current_file)
        
        # Match hunk header: "@@ -246,9 +246,12 @@ def method_name..."
        elif line.startswith('@@') and current_gt:
            # Extract line numbers
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start_line = int(match.group(1))
                num_lines = int(match.group(2)) if match.group(2) else 1
                for lineno in range(start_line, start_line + num_lines):
                    current_gt.lines.add(lineno)
            
            # Extract method name from context
            method_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if method_match:
                current_gt.methods.add(method_match.group(1))
        
        # Look for method definitions in the patch content
        elif line.startswith('+') and current_gt and not line.startswith('+++'):
            method_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if method_match:
                current_gt.methods.add(method_match.group(1))
    
    if current_gt:
        results.append(current_gt)
    
    return results


def run_sbfl_for_task(task: SweTask) -> Tuple[Dict[str, Dict[int, float]], str]:
    """
    Run SBFL for a task and return line scores.
    
    Returns:
        (sbfl_line_scores, error_message)
        sbfl_line_scores: {file_path: {line_num: score}}
    """
    try:
        task.setup_project()
        test_files, ranked_lines, log_file = sbfl.run(task)
        
        # Convert to dict format
        sbfl_line_scores = {}
        for file_path, line_num, score in ranked_lines:
            if file_path not in sbfl_line_scores:
                sbfl_line_scores[file_path] = {}
            sbfl_line_scores[file_path][line_num] = max(
                sbfl_line_scores[file_path].get(line_num, 0), score
            )
        
        return sbfl_line_scores, ""
        
    except Exception as e:
        logger.exception(f"SBFL failed for {task.task_id}: {e}")
        return {}, str(e)


def compute_file_scores(
    sbfl_line_scores: Dict[str, Dict[int, float]],
    issue_stmt: str
) -> Dict[str, float]:
    """
    Compute file-level scores using the improved SBFL scoring.
    Mirrors the logic in ReviewManager._compute_improved_sbfl_scores()
    """
    file_scores = {}
    
    # Extract keywords from issue
    pascal_case = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', issue_stmt)
    snake_case = re.findall(r'\b([a-z_][a-z0-9_]{3,})\b', issue_stmt)
    common_words = {'the', 'this', 'that', 'with', 'from', 'have', 'should', 
                   'would', 'could', 'when', 'where', 'which', 'their', 'there',
                   'about', 'error', 'issue', 'problem', 'function', 'method',
                   'class', 'module', 'file', 'code', 'return', 'value', 'need'}
    keywords = {kw for kw in pascal_case[:10] + snake_case[:15] 
                if kw.lower() not in common_words}
    
    for file_path, line_scores in sbfl_line_scores.items():
        filename = Path(file_path).stem.lower()
        
        # Filter framework files
        file_path_lower = file_path.lower()
        if any(pattern in file_path_lower for pattern in 
               ['conftest', '__init__', 'setup.py', 'test_', '_test.py', '/tests/']):
            continue
        
        if not line_scores:
            continue
        
        # Compute base metrics
        max_score = max(line_scores.values())
        avg_score = sum(line_scores.values()) / len(line_scores)
        num_suspicious = sum(1 for s in line_scores.values() if s > 0.7)
        
        # Size penalty (approximate - we don't have actual file sizes here)
        size_penalty = 1.0
        if len(line_scores) > 200:  # Many lines touched = probably large file
            size_penalty = 0.5
        
        # Base score
        base_score = (max_score * 2.0) + avg_score + (num_suspicious / 100.0)
        base_score *= size_penalty
        
        # Issue keyword boost
        boost = 0.0
        for keyword in keywords:
            kw_lower = keyword.lower()
            if kw_lower in filename or filename in kw_lower:
                boost += 0.5
            elif kw_lower in file_path.lower():
                boost += 0.2
        boost = min(boost, 1.0)
        
        final_score = base_score * (1.0 + boost)
        file_scores[file_path] = final_score
    
    return file_scores


def evaluate_file_localization(
    predicted_file: Optional[str],
    gt_files: List[str]
) -> bool:
    """Check if predicted file matches any ground truth file."""
    if not predicted_file or not gt_files:
        return False
    
    pred = Path(predicted_file).as_posix().lower()
    for gt in gt_files:
        gt_norm = Path(gt).as_posix().lower()
        if pred.endswith(gt_norm) or gt_norm.endswith(pred):
            return True
        # Also check just filename
        if Path(pred).name == Path(gt_norm).name:
            return True
    return False


def evaluate_line_localization(
    predicted_lines: Set[int],
    gt_lines: Set[int],
    tolerance: int = 5
) -> float:
    """
    Compute line recall with tolerance.
    
    Returns recall: what fraction of ground truth lines have a prediction within tolerance.
    """
    if not gt_lines:
        return 0.0
    
    hits = 0
    for gt_line in gt_lines:
        # Check if any predicted line is within tolerance
        for pred_line in predicted_lines:
            if abs(pred_line - gt_line) <= tolerance:
                hits += 1
                break
    
    return hits / len(gt_lines)


def evaluate_task(raw_task: RawSweTask, output_dir: str) -> LocalizationResult:
    """Evaluate localization for a single task."""
    result = LocalizationResult(task_id=raw_task.task_id)
    
    try:
        # Parse ground truth
        gt_patch = raw_task.task_info.get('patch', '')
        if not gt_patch:
            result.error = "No ground truth patch available"
            return result
        
        gt_list = parse_ground_truth_patch(gt_patch)
        result.gt_files = [gt.file_path for gt in gt_list]
        result.gt_lines = {gt.file_path: gt.lines for gt in gt_list}
        result.gt_methods = set()
        for gt in gt_list:
            result.gt_methods.update(gt.methods)
        
        logger.info(f"Ground truth: files={result.gt_files}, methods={result.gt_methods}")
        
        # Convert to SweTask and run SBFL
        task = raw_task.to_task()
        sbfl_line_scores, error = run_sbfl_for_task(task)
        
        if error:
            result.error = f"SBFL failed: {error}"
            return result
        
        if not sbfl_line_scores:
            result.error = "SBFL returned no scores"
            return result
        
        # Store SBFL files
        result.sbfl_files = list(sbfl_line_scores.keys())
        result.predicted_lines = {
            f: set(scores.keys()) for f, scores in sbfl_line_scores.items()
        }
        
        # Compute file-level selection
        issue_stmt = raw_task.task_info.get('problem_statement', '')
        file_scores = compute_file_scores(sbfl_line_scores, issue_stmt)
        
        if file_scores:
            result.predicted_file = max(file_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Predicted file: {result.predicted_file} (score: {file_scores[result.predicted_file]:.2f})")
        
        # Evaluate file-level accuracy
        result.file_correct = evaluate_file_localization(result.predicted_file, result.gt_files)
        
        # Evaluate line-level recall (across all files)
        all_gt_lines = set()
        all_pred_lines = set()
        
        for gt_file, gt_lines in result.gt_lines.items():
            all_gt_lines.update(gt_lines)
            # Find matching predicted file
            for pred_file, pred_lines in result.predicted_lines.items():
                if evaluate_file_localization(pred_file, [gt_file]):
                    all_pred_lines.update(pred_lines)
        
        result.line_recall = evaluate_line_localization(all_pred_lines, all_gt_lines, tolerance=5)
        
        logger.info(f"Results: file_correct={result.file_correct}, line_recall={result.line_recall:.2%}")
        
        # Save SBFL results
        task_output = os.path.join(output_dir, raw_task.task_id)
        os.makedirs(task_output, exist_ok=True)
        
        with open(os.path.join(task_output, 'sbfl_scores.json'), 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable = {
                file: {str(line): score for line, score in scores.items()}
                for file, scores in sbfl_line_scores.items()
            }
            json.dump(serializable, f, indent=2)
        
        with open(os.path.join(task_output, 'file_scores.json'), 'w') as f:
            json.dump(file_scores, f, indent=2)
        
        with open(os.path.join(task_output, 'ground_truth.json'), 'w') as f:
            json.dump({
                'files': result.gt_files,
                'lines': {f: list(lines) for f, lines in result.gt_lines.items()},
                'methods': list(result.gt_methods),
            }, f, indent=2)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error evaluating {raw_task.task_id}: {e}")
        result.error = str(e)
        return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate localization accuracy')
    parser.add_argument('config_file', help='Path to configuration file')
    parser.add_argument('--output-dir', default='localization_eval',
                       help='Output directory for results')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='Maximum number of tasks to evaluate')
    args = parser.parse_args()
    
    # Parse config
    conf = parse_config(args.config_file)
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, 'evaluation.log')
    logger.add(log_file, level='DEBUG')
    
    logger.info(f"Starting localization evaluation")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Output: {output_dir}")
    
    # Load tasks
    tasks = load_tasks(conf)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    logger.info(f"Loaded {len(tasks)} tasks")
    
    # Evaluate each task
    results: List[LocalizationResult] = []
    
    for i, raw_task in enumerate(tasks):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(tasks)}] Evaluating {raw_task.task_id}")
        logger.info(f"{'='*60}")
        
        result = evaluate_task(raw_task, output_dir)
        results.append(result)
        
        # Log progress
        successful = [r for r in results if not r.error]
        if successful:
            file_acc = sum(1 for r in successful if r.file_correct) / len(successful)
            avg_recall = sum(r.line_recall for r in successful) / len(successful)
            logger.info(f"Running stats: file_acc={file_acc:.1%}, line_recall={avg_recall:.1%}")
    
    # Compute summary statistics
    successful_results = [r for r in results if not r.error]
    failed_results = [r for r in results if r.error]
    
    if successful_results:
        file_correct_count = sum(1 for r in successful_results if r.file_correct)
        file_accuracy = file_correct_count / len(successful_results)
        avg_line_recall = sum(r.line_recall for r in successful_results) / len(successful_results)
    else:
        file_accuracy = 0.0
        avg_line_recall = 0.0
    
    # Print summary
    print("\n" + "="*70)
    print("LOCALIZATION EVALUATION SUMMARY")
    print("="*70)
    print(f"Total tasks:        {len(results)}")
    print(f"Successful:         {len(successful_results)}")
    print(f"Failed:             {len(failed_results)}")
    print()
    print(f"FILE-LEVEL ACCURACY: {file_correct_count}/{len(successful_results)} = {file_accuracy:.1%}")
    print(f"LINE RECALL@5:       {avg_line_recall:.1%}")
    print("="*70)
    
    # Per-task breakdown
    print("\nPER-TASK RESULTS:")
    print("-"*70)
    for r in results:
        status = "✅" if r.file_correct else "❌"
        if r.error:
            status = "⚠️"
            print(f"{status} {r.task_id}: ERROR - {r.error}")
        else:
            print(f"{status} {r.task_id}: file={r.file_correct}, line_recall={r.line_recall:.1%}")
    print("-"*70)
    
    # Save detailed results
    summary = {
        "config_file": args.config_file,
        "timestamp": timestamp,
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(failed_results),
        "file_accuracy": file_accuracy,
        "avg_line_recall": avg_line_recall,
        "per_task": [
            {
                "task_id": r.task_id,
                "file_correct": r.file_correct,
                "line_recall": r.line_recall,
                "predicted_file": r.predicted_file,
                "gt_files": r.gt_files,
                "gt_methods": list(r.gt_methods),
                "error": r.error,
            }
            for r in results
        ]
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir}")
    print(f"Summary file: {summary_file}")
    
    # Return exit code based on success rate
    return 0 if file_accuracy >= 0.5 else 1


if __name__ == '__main__':
    sys.exit(main())

