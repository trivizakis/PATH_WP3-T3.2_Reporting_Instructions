# -*- coding: utf-8 -*-
"""
PATH WP3 - Task 3.2: Fairness & Performance Metrics Reporter
============================================================
A self-contained, modular Python class for comprehensive auditing of:
  1. Standard Classification Performance (Accuracy, Precision, Recall, Specificity, F1)
  2. Diagnostic Discrimination (AUROC, Harrell's Concordance Index C-Index)
  3. Probabilistic Calibration (Brier Score, IPA, Expected Calibration Error ECE)
  4. Subgroup Breakdowns (Subgroup Accuracies, C-Indices, Selection Rates)
  5. Multi-Dimensional Fairness Metrics (Accuracy Gap, SPD, DI, EOD, AOD, Theil, C-Index Disparity)

Designed for the PATH Consortium (Horizon Europe) WP3 - Task 3.2 reporting guidelines.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# Optional lifelines support for concordance index; falls back to exact Mann-Whitney / ROC equivalence
try:
    from lifelines.utils import concordance_index as lifelines_c_index
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False


class FairnessMetricsReporter:
    """
    Standardized fairness and discrimination reporter for biomedical machine learning models.

    Parameters
    ----------
    privileged_group : int, default=1
        Value of the sensitive attribute indicating the privileged / reference group.
    unprivileged_group : int, default=0
        Value of the sensitive attribute indicating the unprivileged group.
    sensitive_name : str, default='sensitive_attribute'
        Descriptive label for the protected attribute (e.g. 'scanner_mfs', 'center', 'sex').
    n_bins_ece : int, default=5
        Number of bins for Expected Calibration Error (ECE) calculation.

    Examples
    --------
    >>> reporter = FairnessMetricsReporter()
    >>> reporter.add_evaluation(
    ...     model_name="Logistic_L2",
    ...     variant="Biased",
    ...     y_true=y_test,
    ...     y_pred=y_preds,
    ...     y_prob=y_probs,
    ...     s_true=s_test
    ... )
    >>> df_results = reporter.get_summary_dataframe()
    """

    def __init__(
        self,
        privileged_group: int = 1,
        unprivileged_group: int = 0,
        sensitive_name: str = "sensitive_attribute",
        n_bins_ece: int = 5,
    ):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.sensitive_name = sensitive_name
        self.n_bins_ece = n_bins_ece
        self.records: List[Dict[str, Any]] = []

    # =========================================================================
    # CORE COMPUTATIONAL METHODS
    # =========================================================================
    @staticmethod
    def compute_concordance_index(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Computes Harrell's Concordance Index (C-Index).
        For binary outcomes without censoring, C-Index is equivalent to ROC-AUC.
        Uses lifelines if installed, otherwise exact pair-ranking formulation.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        if len(np.unique(y_true)) < 2:
            return 0.5

        if _HAS_LIFELINES:
            try:
                return float(lifelines_c_index(y_true, y_prob))
            except Exception:
                pass

        try:
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return 0.5

    @staticmethod
    def compute_calibration_metrics(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
    ) -> Dict[str, float]:
        """
        Computes Brier Score, Index of Prediction Accuracy (IPA), and Expected Calibration Error (ECE).

        IPA = 1 - (Brier / Brier_null)
        Where Brier_null is the score of a naive model predicting the dataset base rate.
        """
        y_true = np.asarray(y_true).astype(float)
        y_prob = np.asarray(y_prob).astype(float)

        try:
            brier = float(brier_score_loss(y_true, y_prob))
            null_prob = float(np.mean(y_true))
            brier_null = float(brier_score_loss(y_true, np.full_like(y_prob, null_prob)))
            ipa = float(1.0 - (brier / brier_null)) if brier_null > 0 else 0.0
        except Exception:
            brier = 0.25
            ipa = 0.0

        try:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(y_prob, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            ece = 0.0
            n_total = len(y_true)
            for i in range(n_bins):
                mask = bin_indices == i
                n_bin = np.sum(mask)
                if n_bin > 0:
                    bin_acc = np.mean(y_true[mask])
                    bin_conf = np.mean(y_prob[mask])
                    ece += (n_bin / n_total) * abs(bin_acc - bin_conf)
            ece = float(ece)
        except Exception:
            ece = 0.0

        return {"Brier_Score": brier, "IPA": ipa, "ECE": ece}

    @staticmethod
    def compute_theil_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Theil Index (generalized entropy inequality metric) of model benefit.
        Benefit vector: b_i = y_pred_i - y_true_i + 1
        """
        y_true = np.asarray(y_true).astype(float)
        y_pred = np.asarray(y_pred).astype(float)
        b = y_pred - y_true + 1.0
        mu = np.mean(b)
        if mu <= 0 or np.isnan(mu):
            return 0.0
        ratio = b / mu
        # Avoid log(0)
        ratio = np.where(ratio <= 0, 1e-12, ratio)
        theil = np.mean(ratio * np.log(ratio))
        return float(theil)

    # =========================================================================
    # SINGLE MODEL EVALUATION PIPELINE
    # =========================================================================
    def evaluate_predictions(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
        s_true: Union[np.ndarray, List[int]],
        model_name: str = "Model",
        variant: str = "Standard",
        training_split: str = "Test",
    ) -> Dict[str, Any]:
        """
        Evaluates a complete suite of performance, calibration, subgroup, and fairness metrics
        for a given set of test predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Binary ground truth labels {0, 1}.
        y_pred : array-like of shape (n_samples,)
            Predicted binary class labels {0, 1}.
        y_prob : array-like of shape (n_samples,)
            Continuous predicted probabilities for the positive class [0.0, 1.0].
        s_true : array-like of shape (n_samples,)
            Sensitive attribute values (e.g. 0 = unprivileged, 1 = privileged).
        model_name : str, default='Model'
            Name of model architecture (e.g. 'SGD', 'MLP-16', 'Logistic_L2').
        variant : str, default='Standard'
            Mitigation variant (e.g. 'biased', 'reweighing', 'DIR', 'BGMMOCE').
        training_split : str, default='Test'
            Identifier for data split evaluated.

        Returns
        -------
        dict
            Dictionary of evaluated performance and fairness metrics.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        s_true = np.asarray(s_true).astype(int)

        # 1. Standard Performance Metrics
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # 2. Discrimination & Calibration
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        c_index = self.compute_concordance_index(y_true, y_prob)
        calib = self.compute_calibration_metrics(y_true, y_prob, n_bins=self.n_bins_ece)

        # 3. Subgroup Specific Accuracies and Performance
        mask_0 = s_true == self.unprivileged_group
        mask_1 = s_true == self.privileged_group

        acc_g0 = float(accuracy_score(y_true[mask_0], y_pred[mask_0])) if np.sum(mask_0) > 0 else 0.0
        acc_g1 = float(accuracy_score(y_true[mask_1], y_pred[mask_1])) if np.sum(mask_1) > 0 else 0.0
        acc_gap = float(abs(acc_g1 - acc_g0))

        # Subgroup C-indices
        c_index_g0 = self.compute_concordance_index(y_true[mask_0], y_prob[mask_0])
        c_index_g1 = self.compute_concordance_index(y_true[mask_1], y_prob[mask_1])
        c_index_disp = float(abs(c_index_g1 - c_index_g0))

        # Subgroup selection rates P(y_pred = 1 | s)
        sr_g0 = float(np.mean(y_pred[mask_0] == 1)) if np.sum(mask_0) > 0 else 0.0
        sr_g1 = float(np.mean(y_pred[mask_1] == 1)) if np.sum(mask_1) > 0 else 0.0

        # Subgroup base rates P(y_true = 1 | s)
        br_g0 = float(np.mean(y_true[mask_0] == 1)) if np.sum(mask_0) > 0 else 0.0
        br_g1 = float(np.mean(y_true[mask_1] == 1)) if np.sum(mask_1) > 0 else 0.0

        # True Positive Rates and False Positive Rates per subgroup
        def calc_tpr_fpr(y_t, y_p):
            tn_sub, fp_sub, fn_sub, tp_sub = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            tpr_sub = float(tp_sub / (tp_sub + fn_sub)) if (tp_sub + fn_sub) > 0 else 0.0
            fpr_sub = float(fp_sub / (fp_sub + tn_sub)) if (fp_sub + tn_sub) > 0 else 0.0
            return tpr_sub, fpr_sub

        tpr_g0, fpr_g0 = calc_tpr_fpr(y_true[mask_0], y_pred[mask_0])
        tpr_g1, fpr_g1 = calc_tpr_fpr(y_true[mask_1], y_pred[mask_1])

        # 4. Multi-Dimensional Algorithmic Fairness Metrics
        # Statistical Parity Difference = SR(Privileged) - SR(Unprivileged)
        spd = float(sr_g1 - sr_g0)

        # Disparate Impact = SR(Unprivileged) / SR(Privileged)
        di = float(sr_g0 / sr_g1) if sr_g1 > 0 else (1.0 if sr_g0 == 0 else 0.0)

        # Equal Opportunity Difference = TPR(Privileged) - TPR(Unprivileged)
        eod = float(tpr_g1 - tpr_g0)

        # Average Odds Difference = 0.5 * [(FPR_priv - FPR_unpriv) + (TPR_priv - TPR_unpriv)]
        aod = float(0.5 * ((fpr_g1 - fpr_g0) + (tpr_g1 - tpr_g0)))

        # Theil Index
        theil = self.compute_theil_index(y_true, y_pred)

        record = {
            "Model_Name": model_name,
            "Variant": variant,
            "Training_Split": training_split,
            # Discrimination & Global Performance
            "AUC": auc,
            "C_Index": c_index,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            # Calibration
            "Brier_Score": calib["Brier_Score"],
            "IPA": calib["IPA"],
            "ECE": calib["ECE"],
            # Subgroup Accuracies & Counts
            "Acc_Group_0": acc_g0,
            "Acc_Group_1": acc_g1,
            "N_Group_0": int(np.sum(mask_0)),
            "N_Group_1": int(np.sum(mask_1)),
            "C_Index_Group_0": c_index_g0,
            "C_Index_Group_1": c_index_g1,
            "Selection_Rate_Group_0": sr_g0,
            "Selection_Rate_Group_1": sr_g1,
            "Base_Rate_Group_0": br_g0,
            "Base_Rate_Group_1": br_g1,
            # Fairness Disparity Metrics
            "Accuracy_Gap": acc_gap,
            "Statistical_Parity_Diff": spd,
            "Disparate_Impact": di,
            "Equal_Opportunity_Diff": eod,
            "Average_Odds_Diff": aod,
            "C_Index_Disparity": c_index_disp,
            "Theil_Index": theil,
        }
        return record

    def add_evaluation(
        self,
        model_name: str,
        variant: str,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Union[np.ndarray, List[float]],
        s_true: Union[np.ndarray, List[int]],
        training_split: str = "Test",
    ) -> Dict[str, Any]:
        """
        Computes all metrics and stores the record in internal registry.
        """
        rec = self.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            s_true=s_true,
            model_name=model_name,
            variant=variant,
            training_split=training_split,
        )
        self.records.append(rec)
        return rec

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Returns all registered model evaluations as a clean, standardized pandas DataFrame.
        """
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)

    def to_csv(self, filepath: str) -> None:
        """
        Exports the metrics summary table to CSV.
        """
        df = self.get_summary_dataframe()
        df.to_csv(filepath, index=False)
        print(f"[FairnessMetricsReporter] Successfully exported summary table ({len(df)} models) -> {filepath}")

    def print_summary_table(self) -> str:
        """
        Prints a concise markdown summary table of performance and fairness metrics.
        """
        df = self.get_summary_dataframe()
        if df.empty:
            return "No evaluations registered."

        cols = [
            "Model_Name",
            "Variant",
            "AUC",
            "Accuracy",
            "Accuracy_Gap",
            "Equal_Opportunity_Diff",
            "Statistical_Parity_Diff",
            "C_Index_Disparity",
        ]
        subset = df[cols].copy()
        for c in ["AUC", "Accuracy", "Accuracy_Gap", "Equal_Opportunity_Diff", "Statistical_Parity_Diff", "C_Index_Disparity"]:
            subset[c] = subset[c].apply(lambda x: f"{x:.3f}")

        try:
            md_str = subset.to_markdown(index=False)
        except Exception:
            md_str = subset.to_string(index=False)

        print("\n" + "=" * 90)
        print("PATH WP3 - TASK 3.2: FAIRNESS & PERFORMANCE EVALUATION SUMMARY")
        print("=" * 90)
        print(md_str)
        print("=" * 90 + "\n")
        return md_str

