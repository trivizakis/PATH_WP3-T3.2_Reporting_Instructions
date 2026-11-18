# -*- coding: utf-8 -*-
"""
Synthetic Dataset Generator for Fairness & Bias Auditing
========================================================
Generates reproducible synthetic biomedical/imaging tabular data with
controlled demographic or scanner acquisition bias to benchmark mitigation methods.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_synthetic_fairness_dataset(
    n_samples: int = 1200,
    n_features: int = 20,
    bias_strength: float = 0.35,
    test_size: float = 0.25,
    random_state: int = 42,
    sensitive_attr_name: str = "scanner_field_strength",
    target_name: str = "disease_severity",
):
    """
    Generates a synthetic biomedical tabular dataset with controlled sensitive attribute disparity.

    Parameters
    ----------
    n_samples : int, default=1200
        Total number of patient instances.
    n_features : int, default=20
        Number of continuous clinical/radiomics features.
    bias_strength : float, default=0.35
        Magnitude of correlation injected between sensitive attribute and outcome/features.
    test_size : float, default=0.25
        Proportion of dataset reserved for testing.
    random_state : int, default=42
        Seed for reproducibility.
    sensitive_attr_name : str, default='scanner_field_strength'
        Name of protected attribute (e.g. 0 = 1.5T unprivileged, 1 = 3.0T privileged).
    target_name : str, default='disease_severity'
        Name of binary classification target (0 = benign/indolent, 1 = aggressive/malignant).

    Returns
    -------
    dict:
        X_train, X_test (pd.DataFrame),
        y_train, y_test (np.ndarray),
        s_train, s_test (np.ndarray),
        feature_names (list),
        metadata (dict)
    """
    rng = np.random.RandomState(random_state)

    # 1. Generate sensitive attribute s (binary: 0 = unprivileged / 1.5T, 1 = privileged / 3.0T)
    # Unprivileged group is 40%, privileged is 60%
    s = rng.binomial(n=1, p=0.60, size=n_samples)

    # 2. Latent true clinical disease risk
    latent_risk = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # 3. Add spurious shift correlated with sensitive attribute to create bias
    observed_signal = latent_risk + bias_strength * (s - 0.5)

    # 4. Generate continuous features (e.g., radiomics textures / clinical biomarkers)
    X = np.zeros((n_samples, n_features))
    for j in range(n_features):
        weight = rng.uniform(0.2, 0.8)
        # Features carry both true disease signal and scanner-induced shift
        scanner_artifact = bias_strength * 0.75 * s if j < (n_features // 2) else 0.0
        X[:, j] = weight * observed_signal + scanner_artifact + rng.normal(0, 0.8, size=n_samples)

    feature_cols = [f"radiomics_feat_{i+1:02d}" for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_cols)

    # 5. Generate binary outcome y from logistic probability of latent risk
    logits = 1.2 * latent_risk + 0.4 * bias_strength * s
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= rng.uniform(0, 1, size=n_samples)).astype(int)

    # Split into Train and Test
    indices = np.arange(n_samples)
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = df_features.iloc[idx_train].reset_index(drop=True)
    X_test = df_features.iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_test = y[idx_test]
    s_train = s[idx_train]
    s_test = s[idx_test]

    metadata = {
        "n_samples": n_samples,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "n_features": n_features,
        "bias_strength": bias_strength,
        "sensitive_attr_name": sensitive_attr_name,
        "target_name": target_name,
        "prevalence": float(np.mean(y)),
        "subgroup_distribution_train": {int(g): int(np.sum(s_train == g)) for g in [0, 1]},
        "subgroup_distribution_test": {int(g): int(np.sum(s_test == g)) for g in [0, 1]},
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "s_train": s_train,
        "s_test": s_test,
        "feature_names": feature_cols,
        "metadata": metadata,
    }

