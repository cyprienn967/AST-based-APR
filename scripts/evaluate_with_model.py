#!/usr/bin/env python3
"""
Evaluate localization using a trained LightGBM model.

This script:
1. Loads saved node features from evaluation results
2. Uses the trained model to score/rank nodes
3. Computes localization metrics (file accuracy, method recall, etc.)
4. Compares against the original hand-tuned scoring

Usage:
    python scripts/evaluate_with_model.py local_eval/ --model models/localization_model.pkl
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    
    # Ground truth
    gt_files: List[str] = field(default_factory=list)
    gt_methods: set = field(default_factory=set)
    
    # Original results (hand-tuned weights)
    original_file_correct: bool = False
    original_method_recall: float = 0.0
    original_node_line_recall: float = 0.0
    original_top_methods: List[str] = field(default_factory=list)
    
    # Model results
    model_file_correct: bool = False
    model_method_recall: float = 0.0
    model_node_line_recall: float = 0.0
    model_top_methods: List[str] = field(default_factory=list)
    
    # Rankings
    original_buggy_rank: Optional[int] = None
    model_buggy_rank: Optional[int] = None


def load_model(model_path: str) -> Tuple[Any, List[str]]:
    """Load trained model and feature names."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['feature_names']


def load_task_features(task_dir: Path) -> Tuple[Optional[Dict], List[Dict]]:
    """Load node features and localization results for a task."""
    features_file = task_dir / 'node_features.json'
    localization_file = task_dir / 'localization.json'
    ground_truth_file = task_dir / 'ground_truth.json'
    
    if not features_file.exists():
        return None, []
    
    with open(features_file) as f:
        features_data = json.load(f)
    
    localization = {}
    if localization_file.exists():
        with open(localization_file) as f:
            localization = json.load(f)
    
    ground_truth = {}
    if ground_truth_file.exists():
        with open(ground_truth_file) as f:
            ground_truth = json.load(f)
    
    # Flatten node features
    nodes = []
    for node in features_data.get('nodes', []):
        flat = {
            'task_id': node['task_id'],
            'file_path': node['file_path'],
            'node_id': node['node_id'],
            'node_type': node['node_type'],
            'name': node['name'],
            'start_line': node['start_line'],
            'end_line': node['end_line'],
            'line_count': node['line_count'],
            'file_rank': node.get('file_rank') or 10,
            'is_gt_file': node.get('is_gt_file', False),
            'file_score': node.get('file_score', 0.0),
            
            # Signals
            'sbfl': node['signals']['sbfl'],
            'trace': node['signals']['trace'],
            'slice': node['signals']['slice'],
            'semantic': node['signals']['semantic'],
            'structural': node['signals']['structural'],
            'combined': node['signals']['combined'],
            'boosted': node['signals']['boosted'],
            
            # Negative signals
            'negative_total': node['negative_signals']['total'],
            'neg_pass_only': node['negative_signals']['pass_only_execution'],
            'neg_boilerplate': node['negative_signals']['is_boilerplate'],
            'neg_too_simple': node['negative_signals']['too_simple'],
            'neg_too_complex': node['negative_signals']['too_complex'],
            'neg_except_handler': node['negative_signals']['in_except_handler'],
            'neg_test_debug': node['negative_signals']['is_test_debug'],
            'neg_docstring': node['negative_signals']['docstring_quality'],
            'neg_safe_patterns': node['negative_signals']['safe_patterns'],
            
            # Ground truth
            'is_buggy': node['ground_truth']['is_buggy'],
            'method_match': node['ground_truth']['method_match'],
            'num_buggy_lines': node['ground_truth']['num_buggy_lines'],
        }
        nodes.append(flat)
    
    metadata = {
        'localization': localization,
        'ground_truth': ground_truth,
        'files': features_data.get('files', []),
    }
    
    return metadata, nodes


def evaluate_task(task_dir: Path, model, feature_names: List[str]) -> Optional[TaskResult]:
    """Evaluate a single task with the model."""
    metadata, nodes = load_task_features(task_dir)
    
    if not nodes:
        return None
    
    task_id = task_dir.name
    result = TaskResult(task_id=task_id)
    
    # Get ground truth
    gt = metadata.get('ground_truth', {})
    result.gt_files = gt.get('files', [])
    result.gt_methods = set(gt.get('methods', []))
    
    # Get original results
    loc = metadata.get('localization', {})
    result.original_file_correct = loc.get('file_correct', False)
    result.original_method_recall = loc.get('method_recall', 0.0)
    result.original_node_line_recall = loc.get('node_line_recall', 0.0)
    result.original_top_methods = loc.get('predicted_methods', [])
    
    # Build feature matrix for model
    X = np.array([[n.get(col, 0.0) for col in feature_names] for n in nodes])
    
    # Get model predictions
    model_probs = model.predict_proba(X)[:, 1]
    
    # Get original scores (boosted)
    original_scores = np.array([n['boosted'] for n in nodes])
    
    # Rank by model
    model_ranking = np.argsort(-model_probs)
    original_ranking = np.argsort(-original_scores)
    
    # Find buggy nodes
    buggy_indices = [i for i, n in enumerate(nodes) if n['is_buggy']]
    
    # Find rank of first buggy node
    if buggy_indices:
        for rank, idx in enumerate(model_ranking):
            if idx in buggy_indices:
                result.model_buggy_rank = rank + 1
                break
        
        for rank, idx in enumerate(original_ranking):
            if idx in buggy_indices:
                result.original_buggy_rank = rank + 1
                break
    
    # Get top-5 methods from model ranking
    model_top_5 = model_ranking[:5]
    result.model_top_methods = [nodes[i]['name'] for i in model_top_5 
                                 if nodes[i]['node_type'] in ('FunctionDef', 'AsyncFunctionDef')]
    
    # Calculate model method recall
    if result.gt_methods:
        predicted_methods = set(result.model_top_methods)
        hits = len(predicted_methods & result.gt_methods)
        result.model_method_recall = hits / len(result.gt_methods)
    
    # Check file correctness for model
    # Model predicts within files, so we check if top prediction is in GT file
    if nodes:
        top_node = nodes[model_ranking[0]]
        top_file = top_node['file_path']
        
        # Check if top file matches any GT file (by basename)
        for gt_file in result.gt_files:
            if gt_file in top_file or top_file.endswith(gt_file):
                result.model_file_correct = True
                break
            # Also check basename match
            if Path(top_file).name == Path(gt_file).name:
                result.model_file_correct = True
                break
    
    # Calculate node line recall for model
    if buggy_indices:
        # Get lines covered by top-5 nodes from model
        model_top_5_nodes = [nodes[i] for i in model_ranking[:5]]
        model_lines = set()
        for n in model_top_5_nodes:
            model_lines.update(range(n['start_line'], n['end_line'] + 1))
        
        # Get buggy lines
        buggy_lines = set()
        for i in buggy_indices:
            n = nodes[i]
            buggy_lines.update(range(n['start_line'], n['end_line'] + 1))
        
        if buggy_lines:
            covered = len(model_lines & buggy_lines)
            result.model_node_line_recall = covered / len(buggy_lines)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate localization with trained model')
    parser.add_argument('eval_dirs', nargs='+', help='Directories containing evaluation results')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    if not HAS_NUMPY:
        print("Error: numpy required. Install with: pip install numpy")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from: {args.model}")
    model, feature_names = load_model(args.model)
    print(f"Model uses {len(feature_names)} features")
    
    # Evaluate all tasks
    results = []
    
    for eval_dir in args.eval_dirs:
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            print(f"Warning: {eval_dir} does not exist, skipping")
            continue
        
        task_dirs = [d for d in eval_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        for task_dir in sorted(task_dirs):
            result = evaluate_task(task_dir, model, feature_names)
            if result:
                results.append(result)
    
    if not results:
        print("No results!")
        sys.exit(1)
    
    # Compute aggregate metrics
    n = len(results)
    
    # Original metrics
    orig_file_correct = sum(1 for r in results if r.original_file_correct)
    orig_method_recall = sum(r.original_method_recall for r in results) / n
    orig_node_recall = sum(r.original_node_line_recall for r in results) / n
    
    # Model metrics
    model_file_correct = sum(1 for r in results if r.model_file_correct)
    model_method_recall = sum(r.model_method_recall for r in results) / n
    model_node_recall = sum(r.model_node_line_recall for r in results) / n
    
    # Top-K hit rates
    orig_top1 = sum(1 for r in results if r.original_buggy_rank == 1)
    orig_top3 = sum(1 for r in results if r.original_buggy_rank and r.original_buggy_rank <= 3)
    orig_top5 = sum(1 for r in results if r.original_buggy_rank and r.original_buggy_rank <= 5)
    
    model_top1 = sum(1 for r in results if r.model_buggy_rank == 1)
    model_top3 = sum(1 for r in results if r.model_buggy_rank and r.model_buggy_rank <= 3)
    model_top5 = sum(1 for r in results if r.model_buggy_rank and r.model_buggy_rank <= 5)
    
    # Print comparison
    print("\n" + "="*70)
    print("LOCALIZATION COMPARISON: Original vs LightGBM Model")
    print("="*70)
    print(f"Total tasks evaluated: {n}")
    print()
    
    print(f"{'Metric':<30} {'Original':>15} {'Model':>15} {'Diff':>10}")
    print("-"*70)
    
    def fmt_pct(val, total):
        return f"{val}/{total} ({val/total:.1%})"
    
    def fmt_diff(orig, new):
        diff = new - orig
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1%}"
    
    print(f"{'File accuracy':<30} {fmt_pct(orig_file_correct, n):>15} {fmt_pct(model_file_correct, n):>15} {fmt_diff(orig_file_correct/n, model_file_correct/n):>10}")
    print(f"{'Method recall':<30} {orig_method_recall:>14.1%} {model_method_recall:>14.1%} {fmt_diff(orig_method_recall, model_method_recall):>10}")
    print(f"{'Node line recall':<30} {orig_node_recall:>14.1%} {model_node_recall:>14.1%} {fmt_diff(orig_node_recall, model_node_recall):>10}")
    print()
    print(f"{'Top-1 hit rate':<30} {fmt_pct(orig_top1, n):>15} {fmt_pct(model_top1, n):>15} {fmt_diff(orig_top1/n, model_top1/n):>10}")
    print(f"{'Top-3 hit rate':<30} {fmt_pct(orig_top3, n):>15} {fmt_pct(model_top3, n):>15} {fmt_diff(orig_top3/n, model_top3/n):>10}")
    print(f"{'Top-5 hit rate':<30} {fmt_pct(orig_top5, n):>15} {fmt_pct(model_top5, n):>15} {fmt_diff(orig_top5/n, model_top5/n):>10}")
    print("="*70)
    
    # Show improved/degraded tasks
    improved = [r for r in results if r.model_buggy_rank and r.original_buggy_rank 
                and r.model_buggy_rank < r.original_buggy_rank]
    degraded = [r for r in results if r.model_buggy_rank and r.original_buggy_rank 
                and r.model_buggy_rank > r.original_buggy_rank]
    
    print(f"\nTasks improved: {len(improved)}")
    for r in improved[:5]:
        print(f"  {r.task_id}: rank {r.original_buggy_rank} → {r.model_buggy_rank}")
    
    print(f"\nTasks degraded: {len(degraded)}")
    for r in degraded[:5]:
        print(f"  {r.task_id}: rank {r.original_buggy_rank} → {r.model_buggy_rank}")
    
    # Save detailed results
    if args.output:
        output_data = {
            'summary': {
                'total_tasks': n,
                'original': {
                    'file_accuracy': orig_file_correct / n,
                    'method_recall': orig_method_recall,
                    'node_line_recall': orig_node_recall,
                    'top_1_rate': orig_top1 / n,
                    'top_3_rate': orig_top3 / n,
                    'top_5_rate': orig_top5 / n,
                },
                'model': {
                    'file_accuracy': model_file_correct / n,
                    'method_recall': model_method_recall,
                    'node_line_recall': model_node_recall,
                    'top_1_rate': model_top1 / n,
                    'top_3_rate': model_top3 / n,
                    'top_5_rate': model_top5 / n,
                },
            },
            'per_task': [
                {
                    'task_id': r.task_id,
                    'original_buggy_rank': r.original_buggy_rank,
                    'model_buggy_rank': r.model_buggy_rank,
                    'original_method_recall': r.original_method_recall,
                    'model_method_recall': r.model_method_recall,
                }
                for r in results
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

