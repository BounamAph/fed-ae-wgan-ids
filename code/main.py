"""
main.py
========
Master execution script for AE-WGAN Anomaly-Based Intrusion Detection System.

Based on: Arafah et al. (2025), Applied Soft Computing, 168, 112455.

Full Pipeline:
    1. Preprocessing (Preprocessing.py)
    2. AE-WGAN Oversampling (AE_WGAN.py)
    3. Generative Quality Evaluation (Generative_scores.py)
    4. Classifier Training & Evaluation (Classifiers.py)

Datasets:
    - NSL-KDD (University of New Brunswick)
    - CICIDS-2017 (University of New Brunswick)

Usage:
    python main.py --dataset nslkdd --train_path KDDTrain+.txt --test_path KDDTest+.txt
    python main.py --dataset cicids --data_dir ./CICIDS-2017/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Local imports
from Preprocessing import preprocess_nslkdd, preprocess_cicids, get_attack_data
from AE_WGAN import AEWGAN, oversample_attacks
from Generative_scores import evaluate_generation_quality
from Classifiers import evaluate_all_classifiers


def run_nslkdd_pipeline(train_path, test_path, output_dir='./results_nslkdd',
                        ae_wgan_epochs=100, ae_epochs=50, 
                        classifier_epochs=50, batch_size=32,
                        target_increment=20000, verbose=1):
    """
    Execute the full pipeline on the NSL-KDD dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "#" * 70)
    print("#  AE-WGAN INTRUSION DETECTION SYSTEM — NSL-KDD PIPELINE")
    print("#" * 70)
    
    # ===== STEP 1: PREPROCESSING =====
    print("\n" + "=" * 70)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    (X_train, X_test,
     y_train_cat, y_test_cat,
     y_train_detail, y_test_detail,
     selected_features, prep_objects) = preprocess_nslkdd(train_path, test_path)
    
    input_dim = X_train.shape[1]  # Should be 24
    print(f"\n  Input dimensionality: {input_dim}")
    
    # ===== STEP 2: BASELINE CLASSIFICATION (before oversampling) =====
    print("\n" + "=" * 70)
    print("  STEP 2: BASELINE CLASSIFICATION (without AE-WGAN)")
    print("=" * 70)
    
    # Binary classification baseline
    baseline_binary, _ = evaluate_all_classifiers(
        X_train, y_train_cat, X_test, y_test_cat,
        input_dim=input_dim, dataset='nslkdd', binary=True,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # Multiclass classification baseline
    baseline_multi, _ = evaluate_all_classifiers(
        X_train, y_train_cat, X_test, y_test_cat,
        input_dim=input_dim, dataset='nslkdd', binary=False,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # ===== STEP 3: AE-WGAN OVERSAMPLING =====
    print("\n" + "=" * 70)
    print("  STEP 3: AE-WGAN OVERSAMPLING")
    print("=" * 70)
    
    X_augmented, y_augmented, gen_report, models = oversample_attacks(
        X_train, y_train_detail, input_dim=input_dim,
        target_increment=target_increment,
        epochs=ae_wgan_epochs, ae_epochs=ae_epochs,
        batch_size=batch_size, min_samples=20, verbose=verbose
    )
    
    # Map augmented detailed labels to categories
    from Preprocessing import NSL_KDD_ATTACK_MAP
    y_augmented_cat = y_augmented.map(NSL_KDD_ATTACK_MAP).fillna('Unknown')
    
    # ===== STEP 4: GENERATIVE QUALITY EVALUATION =====
    print("\n" + "=" * 70)
    print("  STEP 4: GENERATIVE QUALITY EVALUATION")
    print("=" * 70)
    
    quality_reports = []
    for attack_type, model in models.items():
        X_real = get_attack_data(X_train, y_train_detail, attack_type)
        if len(X_real) >= 20:
            X_synth = model.generate(min(len(X_real) * 2, 1000))
            report = evaluate_generation_quality(X_real, X_synth, attack_type)
            quality_reports.append(report)
    
    # ===== STEP 5: AE-WGAN ENHANCED CLASSIFICATION =====
    print("\n" + "=" * 70)
    print("  STEP 5: AE-WGAN ENHANCED CLASSIFICATION")
    print("=" * 70)
    
    # Binary classification with augmented data
    enhanced_binary, _ = evaluate_all_classifiers(
        X_augmented, y_augmented_cat, X_test, y_test_cat,
        input_dim=input_dim, dataset='nslkdd', binary=True,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # Multiclass classification with augmented data
    enhanced_multi, _ = evaluate_all_classifiers(
        X_augmented, y_augmented_cat, X_test, y_test_cat,
        input_dim=input_dim, dataset='nslkdd', binary=False,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # ===== STEP 6: COMPARISON =====
    print("\n" + "=" * 70)
    print("  STEP 6: IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    print_comparison(baseline_binary, enhanced_binary, "Binary")
    print_comparison(baseline_multi, enhanced_multi, "Multiclass")
    
    # ===== SAVE RESULTS =====
    save_results(output_dir, {
        'baseline_binary': baseline_binary,
        'baseline_multi': baseline_multi,
        'enhanced_binary': enhanced_binary,
        'enhanced_multi': enhanced_multi,
        'generation_report': gen_report,
        'quality_reports': quality_reports,
        'selected_features': selected_features,
    })
    
    return {
        'baseline_binary': baseline_binary,
        'enhanced_binary': enhanced_binary,
        'baseline_multi': baseline_multi,
        'enhanced_multi': enhanced_multi,
    }


def run_cicids_pipeline(data_dir, output_dir='./results_cicids',
                        ae_wgan_epochs=100, ae_epochs=50,
                        classifier_epochs=50, batch_size=32,
                        target_increment=20000, verbose=1):
    """
    Execute the full pipeline on the CICIDS-2017 dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "#" * 70)
    print("#  AE-WGAN INTRUSION DETECTION SYSTEM — CICIDS-2017 PIPELINE")
    print("#" * 70)
    
    # ===== STEP 1: PREPROCESSING =====
    (X_train, X_test,
     y_train, y_test,
     y_train_binary, y_test_binary,
     selected_features, prep_objects) = preprocess_cicids(data_dir)
    
    input_dim = X_train.shape[1]  # Should be 10
    
    # ===== STEP 2: BASELINE =====
    baseline_binary, _ = evaluate_all_classifiers(
        X_train, y_train_binary, X_test, y_test_binary,
        input_dim=input_dim, dataset='cicids', binary=True,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    baseline_multi, _ = evaluate_all_classifiers(
        X_train, y_train, X_test, y_test,
        input_dim=input_dim, dataset='cicids', binary=False,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # ===== STEP 3: AE-WGAN OVERSAMPLING =====
    X_augmented, y_augmented, gen_report, models = oversample_attacks(
        X_train, y_train, input_dim=input_dim,
        target_increment=target_increment,
        epochs=ae_wgan_epochs, ae_epochs=ae_epochs,
        batch_size=batch_size, min_samples=20, verbose=verbose
    )
    
    y_augmented_binary = y_augmented.apply(
        lambda x: 'Normal' if x == 'BENIGN' else 'Attack'
    )
    
    # ===== STEP 4: QUALITY EVALUATION =====
    quality_reports = []
    for attack_type, model in models.items():
        X_real = get_attack_data(X_train, y_train, attack_type)
        if len(X_real) >= 20:
            X_synth = model.generate(min(len(X_real) * 2, 1000))
            report = evaluate_generation_quality(X_real, X_synth, attack_type)
            quality_reports.append(report)
    
    # ===== STEP 5: ENHANCED CLASSIFICATION =====
    enhanced_binary, _ = evaluate_all_classifiers(
        X_augmented, y_augmented_binary, X_test, y_test_binary,
        input_dim=input_dim, dataset='cicids', binary=True,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    enhanced_multi, _ = evaluate_all_classifiers(
        X_augmented, y_augmented, X_test, y_test,
        input_dim=input_dim, dataset='cicids', binary=False,
        epochs=classifier_epochs, batch_size=batch_size, verbose=verbose
    )
    
    # ===== COMPARISON =====
    print_comparison(baseline_binary, enhanced_binary, "Binary")
    print_comparison(baseline_multi, enhanced_multi, "Multiclass")
    
    save_results(output_dir, {
        'baseline_binary': baseline_binary,
        'baseline_multi': baseline_multi,
        'enhanced_binary': enhanced_binary,
        'enhanced_multi': enhanced_multi,
        'generation_report': gen_report,
        'quality_reports': quality_reports,
        'selected_features': selected_features,
    })


def print_comparison(baseline, enhanced, classification_type):
    """Print comparison between baseline and AE-WGAN enhanced results."""
    print(f"\n  {classification_type} Classification — F1 Score Comparison:")
    print(f"  {'Classifier':<12} {'Baseline':>10} {'AE-WGAN':>10} {'Δ F1':>10}")
    print(f"  {'─' * 44}")
    
    for b, e in zip(baseline, enhanced):
        delta = e['f1_score'] - b['f1_score']
        sign = "+" if delta >= 0 else ""
        print(f"  {b['classifier']:<12} {b['f1_score']:>10.4f} {e['f1_score']:>10.4f} "
              f"{sign}{delta:>9.4f}")


def save_results(output_dir, results):
    """Save results to JSON."""
    filepath = os.path.join(output_dir, 'results.json')
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return obj
    
    serializable = json.loads(json.dumps(results, default=convert))
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n  [Save] Results saved to {filepath}")


# ============================================================================
# QUICK DEMO (small synthetic test)
# ============================================================================

def run_demo():
    """
    Run a quick demo with synthetic data to verify the pipeline works.
    Useful when the actual datasets are not yet downloaded.
    """
    print("\n" + "#" * 70)
    print("#  AE-WGAN DEMO — Synthetic Data Test")
    print("#" * 70)
    
    np.random.seed(42)
    input_dim = 24
    n_train = 2000
    n_test = 500
    
    # Generate synthetic "normal" and "attack" data
    X_normal_train = np.random.randn(n_train // 2, input_dim) * 0.5 + 0.5
    X_attack_train = np.random.randn(n_train // 2, input_dim) * 0.3 + 0.3
    X_train = np.clip(np.vstack([X_normal_train, X_attack_train]), 0, 1)
    
    X_normal_test = np.random.randn(n_test // 2, input_dim) * 0.5 + 0.5
    X_attack_test = np.random.randn(n_test // 2, input_dim) * 0.3 + 0.3
    X_test = np.clip(np.vstack([X_normal_test, X_attack_test]), 0, 1)
    
    y_train_cat = pd.Series(['Normal'] * (n_train // 2) + ['DoS'] * (n_train // 2))
    y_test_cat = pd.Series(['Normal'] * (n_test // 2) + ['DoS'] * (n_test // 2))
    y_train_detail = pd.Series(['normal'] * (n_train // 2) + ['neptune'] * (n_train // 2))
    
    print(f"\n  Synthetic data: Train={X_train.shape}, Test={X_test.shape}")
    
    # --- Test AE-WGAN ---
    print("\n  Testing AE-WGAN on synthetic attack data...")
    model = AEWGAN(input_dim=input_dim, noise_dim=32, latent_dim=16)
    model.train(X_attack_train, epochs=20, ae_epochs=10, verbose=1)
    
    synthetic = model.generate(500)
    print(f"  Generated synthetic data shape: {synthetic.shape}")
    
    # --- Test Generative Scores ---
    print("\n  Evaluating generation quality...")
    report = evaluate_generation_quality(X_attack_train, synthetic, "neptune")
    
    # --- Test Classifiers ---
    print("\n  Testing classifiers...")
    results, _ = evaluate_all_classifiers(
        X_train, y_train_cat, X_test, y_test_cat,
        input_dim=input_dim, dataset='nslkdd', binary=True,
        epochs=10, batch_size=32, verbose=1
    )
    
    print("\n" + "#" * 70)
    print("#  DEMO COMPLETE — All modules working correctly!")
    print("#" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AE-WGAN Intrusion Detection System')
    parser.add_argument('--dataset', type=str, choices=['nslkdd', 'cicids', 'demo'],
                        default='demo', help='Dataset to use')
    parser.add_argument('--train_path', type=str, help='Path to NSL-KDD training file')
    parser.add_argument('--test_path', type=str, help='Path to NSL-KDD testing file')
    parser.add_argument('--data_dir', type=str, help='Path to CICIDS-2017 CSV directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100, help='AE-WGAN training epochs')
    parser.add_argument('--ae_epochs', type=int, default=50, help='AE pre-training epochs')
    parser.add_argument('--clf_epochs', type=int, default=50, help='Classifier training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--target_increment', type=int, default=20000,
                        help='Synthetic samples per attack type')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity (0-2)')
    
    args = parser.parse_args()
    
    if args.dataset == 'demo':
        run_demo()
    elif args.dataset == 'nslkdd':
        if not args.train_path or not args.test_path:
            print("Error: --train_path and --test_path required for NSL-KDD")
            sys.exit(1)
        run_nslkdd_pipeline(
            args.train_path, args.test_path, args.output_dir,
            args.epochs, args.ae_epochs, args.clf_epochs,
            args.batch_size, args.target_increment, args.verbose
        )
    elif args.dataset == 'cicids':
        if not args.data_dir:
            print("Error: --data_dir required for CICIDS-2017")
            sys.exit(1)
        run_cicids_pipeline(
            args.data_dir, args.output_dir,
            args.epochs, args.ae_epochs, args.clf_epochs,
            args.batch_size, args.target_increment, args.verbose
        )
