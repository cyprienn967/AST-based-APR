#!/usr/bin/env python3
"""
Train a LightGBM model to learn localization weights.

This script:
1. Loads node features from evaluation results
2. Trains a LightGBM classifier to predict is_buggy
3. Saves the trained model for use in localization

Usage:
    python scripts/train_lightgbm.py local_eval/ --output models/localization_model.txt
    
    # With multiple eval directories:
    python scripts/train_lightgbm.py local_eval/20251205_190830 local_eval/20251206_000435 --output models/localization_model.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)


# Feature columns to use for training
FEATURE_COLS = [
    # Positive signals
    'sbfl',
    'semantic', 
    'structural',
    # Negative signals (breakdown)
    'negative_total',
    'neg_pass_only',
    'neg_boilerplate',
    'neg_too_simple',
    'neg_too_complex',
    'neg_except_handler',
    'neg_test_debug',
    'neg_docstring',
    'neg_safe_patterns',
    # File-level features
    'file_score',
    'file_rank',
    # Node metadata
    'line_count',
]

# Note: trace and slice are often 0 in offline evaluation, but include if available
OPTIONAL_FEATURE_COLS = ['trace', 'slice']


def load_node_features(eval_dirs: List[str]) -> List[Dict[str, Any]]:
    """Load node features from all task directories."""
    all_features = []
    
    for eval_dir in eval_dirs:
        eval_path = Path(eval_dir)
        if not eval_path.exists():
            print(f"Warning: {eval_dir} does not exist, skipping")
            continue
            
        # Find all task directories
        task_dirs = [d for d in eval_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        for task_dir in sorted(task_dirs):
            features_file = task_dir / 'node_features.json'
            
            if not features_file.exists():
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
                        
                        # File-level metadata
                        'file_rank': node.get('file_rank') or 10,  # Default high rank for GT-only files
                        'is_gt_file': node.get('is_gt_file', False),
                        'file_score': node.get('file_score', 0.0),
                        
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
                
            except Exception as e:
                print(f"Error loading {task_dir.name}: {e}")
    
    return all_features


def prepare_training_data(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Prepare feature matrix X, labels y, and group IDs for training.
    
    Returns:
        X: feature matrix
        y: labels (is_buggy)
        feature_names: list of feature column names
        groups: task IDs for GroupKFold
    """
    # Determine which features to use
    feature_names = []
    for col in FEATURE_COLS:
        if col in features[0]:
            feature_names.append(col)
    
    for col in OPTIONAL_FEATURE_COLS:
        if col in features[0]:
            # Check if any non-zero values
            has_values = any(f.get(col, 0) != 0 for f in features)
            if has_values:
                feature_names.append(col)
    
    print(f"Using {len(feature_names)} features: {feature_names}")
    
    # Build matrices
    X = np.array([[f.get(col, 0.0) for col in feature_names] for f in features])
    y = np.array([1 if f['is_buggy'] else 0 for f in features])
    
    # Task IDs for grouping (to avoid data leakage in cross-validation)
    task_ids = [f['task_id'] for f in features]
    unique_tasks = sorted(set(task_ids))
    task_to_idx = {t: i for i, t in enumerate(unique_tasks)}
    groups = np.array([task_to_idx[t] for t in task_ids])
    
    return X, y, feature_names, groups


def train_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    
    # Handle class imbalance (buggy nodes are rare)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"\nClass distribution: {n_pos} buggy, {n_neg} non-buggy (ratio: 1:{n_neg/n_pos:.1f})")
    
    # LightGBM parameters optimized for small dataset
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 200,
        'verbose': -1,
        'random_state': 42,
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    
    return model


def evaluate_model(model: lgb.LGBMClassifier, X: np.ndarray, y: np.ndarray, 
                   groups: np.ndarray, features: List[Dict], feature_names: List[str]) -> Dict:
    """Evaluate model performance."""
    
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Basic metrics
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.0
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE (on training data)")
    print('='*60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")
    
    # Evaluate localization performance per task
    # For each task: rank nodes by model score, check if buggy node is in top-K
    task_ids = [f['task_id'] for f in features]
    unique_tasks = sorted(set(task_ids))
    
    top_k_hits = {1: 0, 3: 0, 5: 0}
    method_recalls = []
    
    for task_id in unique_tasks:
        # Get indices for this task
        task_mask = np.array([t == task_id for t in task_ids])
        task_probs = y_prob[task_mask]
        task_labels = y[task_mask]
        task_features = [f for i, f in enumerate(features) if task_ids[i] == task_id]
        
        # Sort by model probability (descending)
        sorted_indices = np.argsort(-task_probs)
        
        # Check if any buggy node is in top-K
        for k in [1, 3, 5]:
            top_k_indices = sorted_indices[:k]
            if any(task_labels[i] for i in top_k_indices):
                top_k_hits[k] += 1
        
        # Calculate method recall for this task
        # (what fraction of buggy methods did we find?)
        buggy_methods = set()
        predicted_methods = set()
        
        for i, f in enumerate(task_features):
            if f['is_buggy'] and f['node_type'] in ('FunctionDef', 'AsyncFunctionDef'):
                buggy_methods.add(f['name'])
        
        # Top-5 predicted methods
        top_5_indices = sorted_indices[:5]
        for i in top_5_indices:
            f = task_features[i]
            if f['node_type'] in ('FunctionDef', 'AsyncFunctionDef'):
                predicted_methods.add(f['name'])
        
        if buggy_methods:
            method_recall = len(buggy_methods & predicted_methods) / len(buggy_methods)
            method_recalls.append(method_recall)
    
    n_tasks = len(unique_tasks)
    
    print(f"\n{'='*60}")
    print("LOCALIZATION PERFORMANCE")
    print('='*60)
    print(f"Total tasks: {n_tasks}")
    print(f"Top-1 hit rate: {top_k_hits[1]}/{n_tasks} = {top_k_hits[1]/n_tasks:.1%}")
    print(f"Top-3 hit rate: {top_k_hits[3]}/{n_tasks} = {top_k_hits[3]/n_tasks:.1%}")
    print(f"Top-5 hit rate: {top_k_hits[5]}/{n_tasks} = {top_k_hits[5]/n_tasks:.1%}")
    
    if method_recalls:
        avg_method_recall = sum(method_recalls) / len(method_recalls)
        print(f"Avg method recall (top-5): {avg_method_recall:.1%}")
    
    # Feature importance
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE")
    print('='*60)
    
    importance = model.feature_importances_
    
    sorted_idx = np.argsort(importance)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {feature_names[i]:<20} {importance[i]:>8.1f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'top_1_rate': top_k_hits[1] / n_tasks,
        'top_3_rate': top_k_hits[3] / n_tasks,
        'top_5_rate': top_k_hits[5] / n_tasks,
        'avg_method_recall': sum(method_recalls) / len(method_recalls) if method_recalls else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM localization model')
    parser.add_argument('eval_dirs', nargs='+', help='Directories containing evaluation results')
    parser.add_argument('--output', '-o', default='models/localization_model.txt',
                       help='Output path for trained model')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, do not train')
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        print(f"Error: Missing dependencies. {MISSING_DEP}")
        print("\nInstall with: pip install lightgbm scikit-learn numpy")
        sys.exit(1)
    
    # Load features
    print(f"Loading features from: {args.eval_dirs}")
    features = load_node_features(args.eval_dirs)
    
    if not features:
        print("No features loaded!")
        sys.exit(1)
    
    print(f"\nLoaded {len(features)} nodes from {len(set(f['task_id'] for f in features))} tasks")
    
    # Prepare data
    X, y, feature_names, groups = prepare_training_data(features)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Buggy nodes: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    if args.stats_only:
        print("\n(Stats only mode - not training)")
        return
    
    # Train model
    print("\nTraining LightGBM model...")
    model = train_model(X, y, groups)
    
    # Evaluate
    metrics = evaluate_model(model, X, y, groups, features, feature_names)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as LightGBM native format
    model.booster_.save_model(str(output_path))
    print(f"\nModel saved to: {output_path}")
    
    # Also save as pickle with feature names
    pickle_path = output_path.with_suffix('.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics,
        }, f)
    print(f"Full model saved to: {pickle_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTo use the trained model for localization:")
    print(f"  python scripts/evaluate_with_model.py {args.eval_dirs[0]} --model {pickle_path}")


if __name__ == "__main__":
    main()

