#!/usr/bin/env python3
"""
Merge node features from all tasks into a single CSV/Parquet for LightGBM training.

This script reads node_features.json from each task directory and creates
a unified training dataset with:
- One row per function/method node
- All signal features as columns
- Ground truth label (is_buggy)

Usage:
    python scripts/merge_features_for_lightgbm.py eval_output_dir/ --output features.csv
    python scripts/merge_features_for_lightgbm.py eval_output_dir/ --output features.parquet

Output columns:
    - task_id, file_path, node_id, name, node_type (metadata)
    - sbfl, trace, slice, semantic, structural, combined, boosted (positive signals)
    - negative_total, pass_only_execution, is_boilerplate, too_simple, etc. (negative signals)
    - is_buggy, method_match, num_buggy_lines (ground truth)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. CSV output will use basic format.")


def load_node_features(eval_dir: str) -> List[Dict[str, Any]]:
    """
    Load node features from all task directories.
    
    Returns list of flattened feature dictionaries.
    """
    all_features = []
    task_dirs = [d for d in os.listdir(eval_dir) 
                 if os.path.isdir(os.path.join(eval_dir, d)) 
                 and not d.startswith('.')]
    
    print(f"Found {len(task_dirs)} task directories")
    
    for task_dir in sorted(task_dirs):
        features_file = os.path.join(eval_dir, task_dir, 'node_features.json')
        
        if not os.path.exists(features_file):
            print(f"  Skipping {task_dir}: no node_features.json")
            continue
        
        try:
            with open(features_file) as f:
                data = json.load(f)
            
            nodes = data.get('nodes', [])
            
            for node in nodes:
                # Flatten the nested structure
                flat = {
                    # Metadata
                    'task_id': node['task_id'],
                    'file_path': node['file_path'],
                    'node_id': node['node_id'],
                    'node_type': node['node_type'],
                    'name': node['name'],
                    'start_line': node['start_line'],
                    'end_line': node['end_line'],
                    'line_count': node['line_count'],
                    
                    # Positive signals
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
                
                all_features.append(flat)
            
            buggy_count = sum(1 for n in nodes if n['ground_truth']['is_buggy'])
            print(f"  {task_dir}: {len(nodes)} nodes, {buggy_count} buggy")
            
        except Exception as e:
            print(f"  Error loading {task_dir}: {e}")
    
    return all_features


def save_csv_basic(features: List[Dict], output_path: str):
    """Save features to CSV without pandas."""
    if not features:
        print("No features to save!")
        return
    
    columns = list(features[0].keys())
    
    with open(output_path, 'w') as f:
        # Header
        f.write(','.join(columns) + '\n')
        
        # Rows
        for row in features:
            values = []
            for col in columns:
                val = row[col]
                if isinstance(val, str):
                    # Escape commas and quotes in strings
                    val = '"' + val.replace('"', '""') + '"'
                elif isinstance(val, bool):
                    val = '1' if val else '0'
                else:
                    val = str(val)
                values.append(val)
            f.write(','.join(values) + '\n')


def compute_statistics(features: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    if not features:
        return {}
    
    total = len(features)
    buggy = sum(1 for f in features if f['is_buggy'])
    
    stats = {
        'total_nodes': total,
        'buggy_nodes': buggy,
        'non_buggy_nodes': total - buggy,
        'buggy_ratio': buggy / total if total > 0 else 0,
        'unique_tasks': len(set(f['task_id'] for f in features)),
    }
    
    # Signal statistics
    signal_cols = ['sbfl', 'trace', 'slice', 'semantic', 'structural', 
                   'negative_total', 'combined', 'boosted']
    
    for col in signal_cols:
        values = [f[col] for f in features]
        non_zero = [v for v in values if v != 0]
        
        stats[f'{col}_mean'] = sum(values) / len(values) if values else 0
        stats[f'{col}_nonzero_count'] = len(non_zero)
        stats[f'{col}_nonzero_pct'] = len(non_zero) / len(values) * 100 if values else 0
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Merge node features into LightGBM training dataset'
    )
    parser.add_argument('eval_dir', help='Directory containing evaluation results')
    parser.add_argument('--output', '-o', default='lightgbm_features.csv',
                       help='Output file path (.csv or .parquet)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only print statistics, do not save')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.eval_dir):
        print(f"Error: {args.eval_dir} is not a directory")
        sys.exit(1)
    
    print(f"Loading features from: {args.eval_dir}")
    features = load_node_features(args.eval_dir)
    
    if not features:
        print("No features loaded!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print('='*60)
    
    stats = compute_statistics(features)
    
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Buggy nodes: {stats['buggy_nodes']} ({stats['buggy_ratio']*100:.1f}%)")
    print(f"Non-buggy nodes: {stats['non_buggy_nodes']}")
    print(f"Unique tasks: {stats['unique_tasks']}")
    
    print(f"\nSignal coverage:")
    for signal in ['sbfl', 'semantic', 'structural', 'negative_total']:
        pct = stats.get(f'{signal}_nonzero_pct', 0)
        count = stats.get(f'{signal}_nonzero_count', 0)
        mean = stats.get(f'{signal}_mean', 0)
        print(f"  {signal}: {count} non-zero ({pct:.1f}%), mean={mean:.4f}")
    
    if args.stats_only:
        print("\n(Stats only mode - not saving)")
        return
    
    # Save output
    print(f"\nSaving to: {args.output}")
    
    if args.output.endswith('.parquet'):
        if HAS_PANDAS:
            df = pd.DataFrame(features)
            df.to_parquet(args.output, index=False)
            print(f"Saved {len(features)} rows to {args.output}")
        else:
            print("Error: parquet output requires pandas. Install with: pip install pandas pyarrow")
            sys.exit(1)
    else:
        if HAS_PANDAS:
            df = pd.DataFrame(features)
            df.to_csv(args.output, index=False)
        else:
            save_csv_basic(features, args.output)
        print(f"Saved {len(features)} rows to {args.output}")
    
    # Also save statistics
    stats_file = args.output.rsplit('.', 1)[0] + '_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_file}")
    
    print("\n✅ Done! Ready for LightGBM training.")
    print("\nFeature columns for LightGBM:")
    print("  Positive: sbfl, trace, slice, semantic, structural")
    print("  Negative: negative_total, neg_pass_only, neg_boilerplate, neg_too_simple, etc.")
    print("  Target: is_buggy")


if __name__ == "__main__":
    main()

