#!/usr/bin/env python3
"""
Analyze localization accuracy from EXISTING results (no SBFL re-computation).

Use this on your vanilla-lite results directory if SBFL was already run.

Usage:
    python scripts/analyze_existing_localization.py vanilla-lite

This parses:
- meta.json: contains the ground truth developer patch
- output_*/sbfl_result.json or info.log: contains SBFL scores
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass  
class TaskResult:
    task_id: str
    gt_files: List[str] = field(default_factory=list)
    gt_lines: Dict[str, Set[int]] = field(default_factory=dict)
    gt_methods: Set[str] = field(default_factory=set)
    predicted_file: Optional[str] = None
    sbfl_files: List[str] = field(default_factory=list)
    file_correct: bool = False
    line_recall: float = 0.0
    error: Optional[str] = None


def parse_ground_truth_patch(patch_str: str) -> Dict[str, Set[int]]:
    """Parse developer patch to get file -> lines mapping."""
    results = {}
    current_file = None
    
    for line in patch_str.split('\n'):
        if line.startswith('diff --git'):
            match = re.search(r'b/(.+?)(?:\s|$)', line)
            if match:
                current_file = match.group(1)
                results[current_file] = set()
        
        elif line.startswith('@@') and current_file:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start_line = int(match.group(1))
                num_lines = int(match.group(2)) if match.group(2) else 1
                for lineno in range(start_line, start_line + num_lines):
                    results[current_file].add(lineno)
    
    return results


def extract_methods_from_patch(patch_str: str) -> Set[str]:
    """Extract method names from patch."""
    methods = set()
    # Look for function/method context in hunk headers
    for match in re.finditer(r'@@.*@@\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)', patch_str):
        methods.add(match.group(1))
    # Also look in added lines
    for match in re.finditer(r'^\+.*def\s+([a-zA-Z_][a-zA-Z0-9_]*)', patch_str, re.MULTILINE):
        methods.add(match.group(1))
    return methods


def parse_sbfl_from_log(log_file: str) -> Dict[str, Dict[int, float]]:
    """Try to parse SBFL scores from info.log."""
    scores = {}
    try:
        with open(log_file) as f:
            content = f.read()
        
        # Look for SBFL debug output patterns
        # Pattern: "file.py:line_num score=0.xxx"
        for match in re.finditer(r'([^\s:]+\.py):(\d+)\s+score[=:]?\s*([\d.]+)', content):
            file_path = match.group(1)
            line_num = int(match.group(2))
            score = float(match.group(3))
            
            if file_path not in scores:
                scores[file_path] = {}
            scores[file_path][line_num] = max(scores[file_path].get(line_num, 0), score)
        
        # Also try to find "Selected file X based on" patterns
        file_match = re.search(r'Selected file[:\s]+(\S+\.py)', content)
        if file_match and not scores:
            scores[file_match.group(1)] = {1: 1.0}  # Dummy score
            
    except Exception as e:
        pass
    
    return scores


def compute_file_scores(
    sbfl_line_scores: Dict[str, Dict[int, float]],
    issue_stmt: str
) -> Dict[str, float]:
    """Compute file-level scores using improved SBFL scoring."""
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
        
        max_score = max(line_scores.values())
        avg_score = sum(line_scores.values()) / len(line_scores)
        num_suspicious = sum(1 for s in line_scores.values() if s > 0.7)
        
        size_penalty = 0.5 if len(line_scores) > 200 else 1.0
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
        
        file_scores[file_path] = base_score * (1.0 + boost)
    
    return file_scores


def file_matches(pred: str, gt_list: List[str]) -> bool:
    """Check if predicted file matches any ground truth file."""
    if not pred:
        return False
    pred_norm = Path(pred).as_posix().lower()
    for gt in gt_list:
        gt_norm = Path(gt).as_posix().lower()
        if pred_norm.endswith(gt_norm) or gt_norm.endswith(pred_norm):
            return True
        if Path(pred_norm).name == Path(gt_norm).name:
            return True
    return False


def line_recall(pred_lines: Set[int], gt_lines: Set[int], tolerance: int = 5) -> float:
    """Compute line recall with tolerance."""
    if not gt_lines:
        return 0.0
    hits = 0
    for gt_line in gt_lines:
        for pred_line in pred_lines:
            if abs(pred_line - gt_line) <= tolerance:
                hits += 1
                break
    return hits / len(gt_lines)


def analyze_task_dir(task_dir: Path) -> Optional[TaskResult]:
    """Analyze a single task directory."""
    # Find meta.json
    meta_file = task_dir / 'meta.json'
    if not meta_file.exists():
        return None
    
    try:
        with open(meta_file) as f:
            meta = json.load(f)
    except:
        return None
    
    task_id = meta.get('task_id', task_dir.name.split('_')[0])
    result = TaskResult(task_id=task_id)
    
    # Parse ground truth
    gt_patch = meta.get('task_info', {}).get('patch', '')
    if not gt_patch:
        result.error = "No ground truth patch"
        return result
    
    result.gt_lines = parse_ground_truth_patch(gt_patch)
    result.gt_files = list(result.gt_lines.keys())
    result.gt_methods = extract_methods_from_patch(gt_patch)
    
    # Get issue statement
    issue_stmt = meta.get('task_info', {}).get('problem_statement', '')
    
    # Try to find SBFL scores
    sbfl_scores = {}
    
    # Check output_* directories for sbfl data
    for output_dir in task_dir.glob('output_*'):
        # Try sbfl_result.json
        sbfl_file = output_dir / 'sbfl_result.json'
        if sbfl_file.exists():
            try:
                with open(sbfl_file) as f:
                    data = json.load(f)
                # Assuming format: [[file, line_start, line_end, score], ...]
                for entry in data:
                    if len(entry) >= 4:
                        file_path, line_start, line_end, score = entry[:4]
                        if file_path not in sbfl_scores:
                            sbfl_scores[file_path] = {}
                        for line in range(int(line_start), int(line_end) + 1):
                            sbfl_scores[file_path][line] = max(
                                sbfl_scores[file_path].get(line, 0), float(score)
                            )
            except:
                pass
        
        # Try parsing from info.log
        log_file = output_dir / 'info.log'
        if log_file.exists() and not sbfl_scores:
            sbfl_scores = parse_sbfl_from_log(str(log_file))
    
    # Also check root info.log
    root_log = task_dir / 'info.log'
    if root_log.exists() and not sbfl_scores:
        sbfl_scores = parse_sbfl_from_log(str(root_log))
    
    if not sbfl_scores:
        result.error = "No SBFL scores found"
        return result
    
    result.sbfl_files = list(sbfl_scores.keys())
    
    # Compute file selection
    file_scores = compute_file_scores(sbfl_scores, issue_stmt)
    if file_scores:
        result.predicted_file = max(file_scores.items(), key=lambda x: x[1])[0]
    
    # Evaluate
    result.file_correct = file_matches(result.predicted_file, result.gt_files)
    
    # Line recall
    all_gt_lines = set()
    all_pred_lines = set()
    for gt_file, gt_lines in result.gt_lines.items():
        all_gt_lines.update(gt_lines)
        for pred_file, pred_lines in sbfl_scores.items():
            if file_matches(pred_file, [gt_file]):
                all_pred_lines.update(pred_lines.keys())
    
    result.line_recall = line_recall(all_pred_lines, all_gt_lines)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze existing localization results')
    parser.add_argument('results_dir', help='Directory containing task results')
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        return 1
    
    # Find all task directories
    task_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and (item / 'meta.json').exists():
            task_dirs.append(item)
    
    if not task_dirs:
        # Maybe they're in subdirectories?
        for subdir in results_path.iterdir():
            if subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_dir() and (item / 'meta.json').exists():
                        task_dirs.append(item)
    
    print(f"Found {len(task_dirs)} task directories")
    
    results = []
    for task_dir in sorted(task_dirs):
        result = analyze_task_dir(task_dir)
        if result:
            results.append(result)
            print(f"  {result.task_id}: file={'✅' if result.file_correct else '❌'}, "
                  f"recall={result.line_recall:.1%}" + (f" (error: {result.error})" if result.error else ""))
    
    # Summary
    successful = [r for r in results if not r.error]
    
    # Calculate metrics (define before use)
    file_acc = 0.0
    avg_recall = 0.0
    if successful:
        file_acc = sum(1 for r in successful if r.file_correct) / len(successful)
        avg_recall = sum(r.line_recall for r in successful) / len(successful)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful)}")
    
    if successful:
        print(f"\nFILE ACCURACY: {file_acc:.1%}")
        print(f"LINE RECALL@5: {avg_recall:.1%}")
    else:
        print("\nNo successful evaluations - cannot compute metrics")
    
    print("="*60)
    
    # Save results
    output_file = results_path / 'localization_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total': len(results),
            'successful': len(successful),
            'file_accuracy': file_acc,
            'avg_line_recall': avg_recall,
            'tasks': [
                {
                    'task_id': r.task_id,
                    'file_correct': r.file_correct,
                    'line_recall': r.line_recall,
                    'predicted_file': r.predicted_file,
                    'gt_files': r.gt_files,
                    'gt_methods': list(r.gt_methods),
                    'error': r.error,
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

