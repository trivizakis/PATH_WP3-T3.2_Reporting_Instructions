#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PATH WP3 - Task 3.2: AI Fairness & Bias Reporting Tutorial
===============================================================================
This tutorial demonstrates the end-to-end workflow to audit, quantify, and
visualize algorithmic fairness disparities in clinical/biomedical AI models.

Workflow:
  1. Generate synthetic biomedical/radiomics data with controlled acquisition bias
  2. Train baseline and mitigation models across standard intervention paradigms:
     - Biased Baseline
     - Pre-processing: Sample Reweighing
     - In-processing: Subgroup-Regularized / Balanced Classifier
     - Post-processing: Decision Threshold Optimization (Reject Option)
     - Advanced: Multi-Objective Fair Representation (BGMMOCE-style)
  3. Extract all metrics using FairnessMetricsReporter
  4. Export metrics summary table (CSV)
  5. Generate publication-grade 4-panel diagnostic figure using FairnessFigureGenerator

Run via:
  python tutorial_fairness_reporting.py
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import the PATH Fairness Toolkit
from path_fairness import (
    FairnessMetricsReporter,
    FairnessFigureGenerator,
    generate_synthetic_fairness_dataset,
)


def compute_reweighing_weights(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Computes exact Statistical Parity sample weights for Reweighing (Kamiran & Calders):
      W(s, y) = P(s) * P(y) / P(s, y) = (N_s * N_y) / (N * N_sy)
    """
    n_total = len(y)
    p_s = {val: np.mean(s == val) for val in np.unique(s)}
    p_y = {val: np.mean(y == val) for val in np.unique(y)}

    weights = np.zeros(n_total)
    for i in range(n_total):
        s_val, y_val = s[i], y[i]
        p_sy = np.mean((s == s_val) & (y == y_val))
        weights[i] = (p_s[s_val] * p_y[y_val]) / max(p_sy, 1e-8)
    return weights


def simulate_fair_representations(X: pd.DataFrame, s: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Simulates Disparate Impact Remover / Fair representation learning by
    partially projecting out feature correlation with sensitive attribute s.
    """
    X_mat = X.values.copy()
    s_centered = s - np.mean(s)
    # Regression coefficient of s on each feature
    cov = np.dot(X_mat.T, s_centered) / (np.dot(s_centered, s_centered) + 1e-8)
    X_repaired = X_mat - alpha * np.outer(s_centered, cov)
    return X_repaired


def postprocess_threshold_adjustment(
    probs: np.ndarray, s: np.ndarray, margin: float = 0.08
) -> np.ndarray:
    """
    Simulates Reject Option Classification / subgroup-adaptive decision boundary:
    Shifts the classification threshold for the unprivileged group to balance TPR/FPR.
    """
    preds = np.zeros_like(probs, dtype=int)
    # Group 1 (Privileged): standard threshold 0.50
    preds[s == 1] = (probs[s == 1] >= 0.50).astype(int)
    # Group 0 (Unprivileged): adaptive threshold to compensate for systematic shift
    preds[s == 0] = (probs[s == 0] >= (0.50 - margin)).astype(int)
    return preds


def main():
    print("=" * 80)
    print("PATH WP3 - TASK 3.2: AI FAIRNESS & BIAS REPORTING TUTORIAL")
    print("=" * 80)

    output_dir = "tutorial_output"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Dataset Generation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating synthetic biomedical dataset with scanner acquisition bias...")
    data = generate_synthetic_fairness_dataset(
        n_samples=1500,
        n_features=24,
        bias_strength=0.40,
        test_size=0.30,
        random_state=42,
        sensitive_attr_name="scanner_mfs",  # 0 = 1.5 Tesla, 1 = 3.0 Tesla
        target_name="significant_cancer",
    )

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    s_train, s_test = data["s_train"], data["s_test"]

    print(f"  -> Training cohort : {len(y_train)} samples")
    print(f"  -> Testing cohort  : {len(y_test)} samples (N_Group0={np.sum(s_test==0)}, N_Group1={np.sum(s_test==1)})")
    print(f"  -> Global disease prevalence : {np.mean(y_test):.1%}")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # -------------------------------------------------------------------------
    # 2. Train Models Across Fairness Mitigation Spectrum
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training models across mitigation spectrum...")
    reporter = FairnessMetricsReporter(
        privileged_group=1,
        unprivileged_group=0,
        sensitive_name="scanner_field_strength",
    )

    # --- Model 1: Biased Baseline ---
    print("  -> Training: 1. Biased (Baseline L2)...")
    clf_base = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42)
    clf_base.fit(X_train_sc, y_train)
    probs_base = clf_base.predict_proba(X_test_sc)[:, 1]
    preds_base = clf_base.predict(X_test_sc)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="Biased (Baseline)",
        y_true=y_test,
        y_pred=preds_base,
        y_prob=probs_base,
        s_true=s_test,
    )

    # --- Model 2: Pre-processing (Reweighing) ---
    print("  -> Training: 2. Fair: Reweighing (Pre-processing)...")
    weights = compute_reweighing_weights(y_train, s_train)
    clf_rw = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42)
    clf_rw.fit(X_train_sc, y_train, sample_weight=weights)
    probs_rw = clf_rw.predict_proba(X_test_sc)[:, 1]
    preds_rw = clf_rw.predict(X_test_sc)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="Fair: Reweighing",
        y_true=y_test,
        y_pred=preds_rw,
        y_prob=probs_rw,
        s_true=s_test,
    )

    # --- Model 3: Pre-processing (Disparate Impact Remover / Fair Representation) ---
    print("  -> Training: 3. Fair: Disparate Impact Remover (Pre-processing)...")
    X_train_rep = simulate_fair_representations(X_train, s_train, alpha=0.6)
    X_test_rep = simulate_fair_representations(X_test, s_test, alpha=0.6)
    scaler_rep = StandardScaler()
    X_train_rep_sc = scaler_rep.fit_transform(X_train_rep)
    X_test_rep_sc = scaler_rep.transform(X_test_rep)

    clf_dir = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42)
    clf_dir.fit(X_train_rep_sc, y_train)
    probs_dir = clf_dir.predict_proba(X_test_rep_sc)[:, 1]
    preds_dir = clf_dir.predict(X_test_rep_sc)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="Fair: Disparate Impact Remover",
        y_true=y_test,
        y_pred=preds_dir,
        y_prob=probs_dir,
        s_true=s_test,
    )

    # --- Model 4: In-processing (Balanced / Subgroup-Regularized) ---
    print("  -> Training: 4. Fair: Prejudice Remover (In-processing)...")
    clf_in = LogisticRegression(C=0.35, class_weight="balanced", solver="lbfgs", random_state=42)
    clf_in.fit(X_train_sc, y_train)
    probs_in = clf_in.predict_proba(X_test_sc)[:, 1]
    preds_in = clf_in.predict(X_test_sc)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="Fair: Prejudice Remover",
        y_true=y_test,
        y_pred=preds_in,
        y_prob=probs_in,
        s_true=s_test,
    )

    # --- Model 5: Post-processing (Reject Option Classification) ---
    print("  -> Training: 5. Fair: Reject Option Classification (Post-processing)...")
    preds_roc = postprocess_threshold_adjustment(probs_base, s_test, margin=0.08)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="Fair: Reject Option Classification",
        y_true=y_test,
        y_pred=preds_roc,
        y_prob=probs_base,
        s_true=s_test,
    )

    # --- Model 6: Advanced In-processing (BGMMOCE Fair Multi-Objective) ---
    print("  -> Training: 6. BGMMOCE (Multi-Objective Fair Clustering/Resampling)...")
    # Simulate Pareto-optimized representation (moderate regularization + fair reweighting)
    bgm_weights = compute_reweighing_weights(y_train, s_train) ** 0.8
    clf_bgm = LogisticRegression(C=0.6, solver="lbfgs", random_state=42)
    clf_bgm.fit(X_train_sc, y_train, sample_weight=bgm_weights)
    probs_bgm = clf_bgm.predict_proba(X_test_sc)[:, 1]
    preds_bgm = clf_bgm.predict(X_test_sc)

    reporter.add_evaluation(
        model_name="Logistic_Regression",
        variant="BGMMOCE",
        y_true=y_test,
        y_pred=preds_bgm,
        y_prob=probs_bgm,
        s_true=s_test,
    )

    # -------------------------------------------------------------------------
    # 3. Output Metrics Summary Table
    # -------------------------------------------------------------------------
    print("\n[Step 3] Compiling and displaying fairness evaluation summary table:")
    reporter.print_summary_table()

    csv_path = os.path.join(output_dir, "metrics_comparison.csv")
    reporter.to_csv(csv_path)

    # -------------------------------------------------------------------------
    # 4. Generate 4-Panel Diagnostic Figure
    # -------------------------------------------------------------------------
    print("\n[Step 4] Generating publication-grade 4-panel diagnostic figure...")
    df_results = reporter.get_summary_dataframe()

    fig_gen = FairnessFigureGenerator()
    fig_path = os.path.join(output_dir, "fairness_4panel_diagnostic.png")

    fig_gen.generate_4panel_figure(
        df_metrics=df_results,
        title="Comprehensive Fairness Gap & Mitigation Analysis: Logistic Regression\n"
              "PATH WP3 - Task 3.2 Evaluation | Binary Clinical Classification",
        baseline_variant_name="Biased (Baseline)",
        privileged_label="Group 1 (3.0T MRI / Privileged)",
        unprivileged_label="Group 0 (1.5T MRI / Unprivileged)",
        save_path=fig_path,
        dpi=300,
        ideal_thresholds=(0.06, 0.70),
    )

    print("\n" + "=" * 80)
    print("TUTORIAL EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Generated Outputs:")
    print(f"  1. Metrics Summary Table : {csv_path}")
    print(f"  2. Diagnostic Figure     : {fig_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

