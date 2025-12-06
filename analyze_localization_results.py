#!/usr/bin/env python3
"""Analyze localization test results from local_eval directory."""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

def analyze_localization_results(base_dir: str = "local_eval") -> Dict:
    """Analyze all localization results and compute aggregate metrics."""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return {"error": f"Directory {base_dir} does not exist"}
    
    all_results = []
    per_run_stats = defaultdict(lambda: {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "file_correct_count": 0,
        "method_recalls": [],
        "node_line_recalls": [],
        "line_recalls": [],
        "tasks": []
    })
    
    # Find all localization.json files
    for localization_file in base_path.rglob("localization.json"):
        run_dir = localization_file.parent.parent.name
        task_dir = localization_file.parent.name
        
        try:
            with open(localization_file, 'r') as f:
                data = json.load(f)
            
            # Check if this is a valid result (has metrics)
            has_metrics = any(key in data for key in ['file_correct', 'method_recall', 'node_line_recall'])
            
            if has_metrics:
                result = {
                    "run": run_dir,
                    "task": task_dir,
                    "file_correct": data.get("file_correct", False),
                    "method_recall": data.get("method_recall", 0.0),
                    "node_line_recall": data.get("node_line_recall", 0.0),
                    "line_recall": data.get("line_recall", 0.0),
                    "error": data.get("error", None)
                }
                
                all_results.append(result)
                stats = per_run_stats[run_dir]
                stats["total"] += 1
                
                if result["error"]:
                    stats["failed"] += 1
                else:
                    stats["successful"] += 1
                    if result["file_correct"]:
                        stats["file_correct_count"] += 1
                    stats["method_recalls"].append(result["method_recall"])
                    stats["node_line_recalls"].append(result["node_line_recall"])
                    if result["line_recall"] > 0:
                        stats["line_recalls"].append(result["line_recall"])
                
                stats["tasks"].append(result)
        except Exception as e:
            print(f"Error reading {localization_file}: {e}")
            continue
    
    # Compute aggregate metrics
    total_tasks = len(all_results)
    successful_tasks = sum(1 for r in all_results if not r.get("error"))
    failed_tasks = total_tasks - successful_tasks
    
    successful_results = [r for r in all_results if not r.get("error")]
    
    if successful_results:
        file_correct_count = sum(1 for r in successful_results if r["file_correct"])
        file_accuracy = file_correct_count / len(successful_results)
        avg_method_recall = sum(r["method_recall"] for r in successful_results) / len(successful_results)
        avg_node_line_recall = sum(r["node_line_recall"] for r in successful_results) / len(successful_results)
        
        line_recalls = [r["line_recall"] for r in successful_results if r["line_recall"] > 0]
        avg_line_recall = sum(line_recalls) / len(line_recalls) if line_recalls else 0.0
    else:
        file_accuracy = 0.0
        avg_method_recall = 0.0
        avg_node_line_recall = 0.0
        avg_line_recall = 0.0
        file_correct_count = 0
    
    # Compute per-run metrics
    per_run_summary = {}
    for run, stats in per_run_stats.items():
        if stats["successful"] > 0:
            per_run_summary[run] = {
                "total_tasks": stats["total"],
                "successful_tasks": stats["successful"],
                "failed_tasks": stats["failed"],
                "file_accuracy": stats["file_correct_count"] / stats["successful"],
                "avg_method_recall": sum(stats["method_recalls"]) / len(stats["method_recalls"]) if stats["method_recalls"] else 0.0,
                "avg_node_line_recall": sum(stats["node_line_recalls"]) / len(stats["node_line_recalls"]) if stats["node_line_recalls"] else 0.0,
                "avg_line_recall": sum(stats["line_recalls"]) / len(stats["line_recalls"]) if stats["line_recalls"] else 0.0,
            }
        else:
            per_run_summary[run] = {
                "total_tasks": stats["total"],
                "successful_tasks": 0,
                "failed_tasks": stats["failed"],
                "file_accuracy": 0.0,
                "avg_method_recall": 0.0,
                "avg_node_line_recall": 0.0,
                "avg_line_recall": 0.0,
            }
    
    return {
        "overall": {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "file_accuracy": file_accuracy,
            "file_correct_count": file_correct_count,
            "avg_method_recall": avg_method_recall,
            "avg_node_line_recall": avg_node_line_recall,
            "avg_line_recall": avg_line_recall,
        },
        "per_run": per_run_summary,
        "all_results": all_results
    }

def print_summary(results: Dict):
    """Print a formatted summary of the results."""
    overall = results["overall"]
    
    print("\n" + "="*80)
    print("LOCALIZATION TEST RESULTS SUMMARY")
    print("="*80)
    print(f"\nOVERALL METRICS:")
    print(f"  Total Tasks:           {overall['total_tasks']}")
    print(f"  Successful:            {overall['successful_tasks']} ({overall['success_rate']:.1%})")
    print(f"  Failed:                {overall['failed_tasks']}")
    print()
    print(f"FILE-LEVEL METRICS:")
    print(f"  File Accuracy:         {overall['file_correct_count']}/{overall['successful_tasks']} = {overall['file_accuracy']:.1%}")
    print()
    print(f"NODE-LEVEL METRICS:")
    print(f"  Method Recall:         {overall['avg_method_recall']:.1%}")
    print(f"  Node Line Recall:      {overall['avg_node_line_recall']:.1%}")
    if overall['avg_line_recall'] > 0:
        print(f"  SBFL Line Recall:      {overall['avg_line_recall']:.1%}")
    print("="*80)
    
    print(f"\nPER-RUN BREAKDOWN:")
    print("-"*80)
    print(f"{'Run':<20} {'Total':<8} {'Success':<8} {'Failed':<8} {'File Acc':<10} {'Method R':<10} {'Node R':<10}")
    print("-"*80)
    
    for run, stats in sorted(results["per_run"].items()):
        print(f"{run:<20} {stats['total_tasks']:<8} {stats['successful_tasks']:<8} {stats['failed_tasks']:<8} "
              f"{stats['file_accuracy']:>8.1%} {stats['avg_method_recall']:>8.1%} {stats['avg_node_line_recall']:>8.1%}")
    
    print("-"*80)

if __name__ == "__main__":
    results = analyze_localization_results()
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print_summary(results)
        
        # Save detailed results
        output_file = "localization_metrics_summary.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

