# -*- coding: utf-8 -*-
"""
PATH WP3 - Task 3.2: Comprehensive Fairness Visualizer
======================================================
Produces publication-grade, 4-panel multi-dimensional fairness diagnostic figures:
  Panel A: Accuracy Gap Reduction (|Group 1 - Group 0|)
  Panel B: Subgroup Accuracies (Privileged vs. Unprivileged)
  Panel C: Multidimensional Fairness Metrics (|EOD|, |SPD|, |AOD|, C-Index Disparity)
  Panel D: Performance vs. Fairness Trade-Off Frontier (Pareto Frontier & Collision-Free Annotations)

Adheres strictly to the PATH WP3 - Task 3.2 standardized reporting guidelines.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Default PATH WP3 Color Palette for Mitigation Strategies
DEFAULT_COLOR_PALETTE = {
    "Biased (Baseline)": "#d62728",                  # Red
    "Biased": "#d62728",                             # Red
    "Biased Baseline": "#d62728",                    # Red
    "Fair: Reweighing": "#2ca02c",                   # Green
    "Reweighing": "#2ca02c",                         # Green
    "Fair: Disparate Impact Remover": "#9467bd",     # Purple
    "Disparate Impact Remover": "#9467bd",           # Purple
    "DIR": "#9467bd",                                # Purple
    "Fair: Prejudice Remover": "#17becf",            # Teal / Cyan
    "Prejudice Remover": "#17becf",                  # Teal / Cyan
    "In-processing (Fair)": "#17becf",               # Teal / Cyan
    "Fair: Calibrated Equalized Odds": "#e377c2",    # Pink / Magenta
    "Calibrated Equalized Odds": "#e377c2",          # Pink / Magenta
    "Fair: Equalized Odds": "#ff7f0e",               # Orange
    "Equalized Odds": "#ff7f0e",                     # Orange
    "Fair: Reject Option Classification": "#bcbd22", # Olive / Gold
    "Reject Option Classification": "#bcbd22",       # Olive / Gold
    "ROC": "#bcbd22",                                # Olive / Gold
    "Fair: Reductions (ExpGrad EO)": "#8c564b",      # Brown
    "ExpGrad EO": "#8c564b",                         # Brown
    "Fair: Reductions (GridSearch EO)": "#c49c94",   # Light Brown / Tan
    "GridSearch EO": "#c49c94",                      # Light Brown / Tan
    "BGMMOCE": "#1f77b4",                            # Primary Blue
    "BGMMOCE (Fair Multi-Objective)": "#1f77b4",     # Primary Blue
}

FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


class FairnessFigureGenerator:
    """
    Generator for standardized 4-panel publication-grade fairness figures.

    Parameters
    ----------
    palette : dict, optional
        Custom dictionary mapping variant/model names to hex colors.
    """

    def __init__(self, palette: Optional[Dict[str, str]] = None):
        self.palette = DEFAULT_COLOR_PALETTE.copy()
        if palette:
            self.palette.update(palette)

    def _resolve_color(self, name: str, idx: int) -> str:
        """Resolves color from palette or assigns fallback."""
        if name in self.palette:
            return self.palette[name]
        for key, col in self.palette.items():
            if key.lower() in name.lower() or name.lower() in key.lower():
                return col
        return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

    @staticmethod
    def _compute_pareto_frontier(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Pareto-optimal frontier where:
        x is minimized (Fairness Gap, Lower is Better)
        y is maximized (AUROC, Higher is Better)
        """
        # Sort primarily by x ascending, then y descending
        sorted_indices = np.lexsort((-ys, xs))
        xs_sorted = xs[sorted_indices]
        ys_sorted = ys[sorted_indices]

        pareto_x = []
        pareto_y = []
        current_max_y = -np.inf

        for x_val, y_val in zip(xs_sorted, ys_sorted):
            if y_val > current_max_y:
                pareto_x.append(x_val)
                pareto_y.append(y_val)
                current_max_y = y_val

        return np.array(pareto_x), np.array(pareto_y)

    @staticmethod
    def _place_smart_labels(
        points: List[Tuple[float, float, str]],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> List[Tuple[float, float, float, float, str]]:
        """
        Dynamic collision-avoidance label placement algorithm using penalty minimization.
        Returns: list of (data_x, data_y, text_x, text_y, label)
        """
        placed_boxes = []  # (xmin, xmax, ymin, ymax)
        result = []

        xr = max(1e-5, x_range[1] - x_range[0])
        yr = max(1e-5, y_range[1] - y_range[0])

        # Candidate directional offset fractions (dx_frac, dy_frac)
        candidate_offsets = [
            (0.000, 0.038),
            (0.000, -0.038),
            (0.065, 0.028),
            (-0.065, 0.028),
            (0.065, -0.028),
            (-0.065, -0.028),
            (0.085, 0.000),
            (-0.085, 0.000),
            (0.000, 0.062),
            (0.000, -0.062),
            (0.115, 0.025),
            (-0.115, 0.025),
            (0.115, -0.025),
            (-0.115, -0.025),
            (0.140, 0.000),
            (-0.140, 0.000),
        ]

        # Reserve top-left region for the "Ideal Region" badge
        ideal_xmin = x_range[0]
        ideal_xmax = x_range[0] + 0.28 * xr
        ideal_ymin = y_range[1] - 0.16 * yr
        ideal_ymax = y_range[1]
        placed_boxes.append((ideal_xmin, ideal_xmax, ideal_ymin, ideal_ymax))

        box_w = 0.135 * xr
        box_h = 0.052 * yr

        for x, y, lbl in points:
            best_placement = None
            min_penalty = 1e9

            for frac_x, frac_y in candidate_offsets:
                tx = x + frac_x * xr
                ty = y + frac_y * yr

                # Boundary containment constraints
                if tx - box_w / 2 < x_range[0] + 0.005 * xr or tx + box_w / 2 > x_range[1] - 0.005 * xr:
                    continue
                if ty - box_h / 2 < y_range[0] + 0.015 * yr or ty + box_h / 2 > y_range[1] - 0.015 * yr:
                    continue

                b_xmin, b_xmax = tx - box_w / 2, tx + box_w / 2
                b_ymin, b_ymax = ty - box_h / 2, ty + box_h / 2

                # Overlap penalty
                overlap_penalty = 0.0
                for ex_xmin, ex_xmax, ex_ymin, ex_ymax in placed_boxes:
                    ox = max(0.0, min(b_xmax, ex_xmax) - max(b_xmin, ex_xmin))
                    oy = max(0.0, min(b_ymax, ex_ymax) - max(b_ymin, ex_ymin))
                    overlap_penalty += (ox * oy) / (box_w * box_h)

                dist_penalty = (frac_x**2 + frac_y**2) * 8.0
                penalty = overlap_penalty * 2000.0 + dist_penalty

                if penalty < min_penalty:
                    min_penalty = penalty
                    best_placement = (tx, ty, b_xmin, b_xmax, b_ymin, b_ymax)

            if best_placement:
                tx, ty, b_xmin, b_xmax, b_ymin, b_ymax = best_placement
                placed_boxes.append((b_xmin, b_xmax, b_ymin, b_ymax))
                result.append((x, y, tx, ty, lbl))
            else:
                result.append((x, y, x + 0.01 * xr, y + 0.02 * yr, lbl))

        return result

    def generate_4panel_figure(
        self,
        df_metrics: pd.DataFrame,
        title: str = "Comprehensive Fairness Gap & Mitigation Analysis",
        baseline_variant_name: str = "biased",
        privileged_label: str = "Group 1 (Privileged)",
        unprivileged_label: str = "Group 0 (Unprivileged)",
        save_path: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[float, float] = (16.0, 12.8),
        ideal_thresholds: Tuple[float, float] = (0.10, 0.70),
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Constructs and renders the standardized 4-panel diagnostic figure.

        Parameters
        ----------
        df_metrics : pd.DataFrame
            Summary DataFrame produced by FairnessMetricsReporter.
        title : str
            Figure super-title.
        baseline_variant_name : str, default='biased'
            Variant name used to identify the biased baseline reference line.
        privileged_label : str
            Display name for Group 1 in Panel B.
        unprivileged_label : str
            Display name for Group 0 in Panel B.
        save_path : str, optional
            Path to write the rendered figure (PNG/PDF).
        dpi : int, default=300
            Resolution for raster output.
        figsize : tuple, default=(16.0, 12.8)
            Figure dimensions in inches.
        ideal_thresholds : tuple, default=(0.10, 0.70)
            (Max acceptable Gap, Min desired AUROC) defining the Ideal Region.

        Returns
        -------
        fig, axes
            Matplotlib Figure and 2x2 ndarray of Axes.
        """
        df = df_metrics.copy()
        n_models = len(df)
        if n_models == 0:
            raise ValueError("df_metrics is empty. Cannot generate figure.")

        # Ensure label formatting
        if "Short_Label" not in df.columns:
            df["Short_Label"] = df.apply(
                lambda r: f"{r.get('Model_Name', 'Model')}\n({r.get('Variant', 'Standard')})", axis=1
            )
        if "Color" not in df.columns:
            df["Color"] = [
                self._resolve_color(str(row.get("Variant", row.get("Model_Name", ""))), i)
                for i, (_, row) in enumerate(df.iterrows())
            ]

        # Determine biased baseline gap reference if available
        baseline_gap = None
        mask_base = df["Variant"].str.lower() == baseline_variant_name.lower()
        if mask_base.any():
            baseline_gap = float(df.loc[mask_base, "Accuracy_Gap"].iloc[0])

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # ---------------------------------------------------------------------
        # Panel A: Accuracy Gap Reduction (|Group 1 - Group 0|)
        # ---------------------------------------------------------------------
        ax00 = axes[0, 0]
        bar_width = max(0.35, min(0.65, 7.5 / n_models))
        bars00 = ax00.bar(
            range(n_models),
            df["Accuracy_Gap"],
            color=df["Color"],
            edgecolor="black",
            linewidth=1.1,
            alpha=0.9,
            width=bar_width,
        )

        for bar, gap in zip(bars00, df["Accuracy_Gap"]):
            font_sz = 9.0 if n_models <= 6 else (8.0 if n_models <= 10 else 7.2)
            ax00.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{gap:.3f}",
                ha="center",
                va="bottom",
                fontsize=font_sz,
                fontweight="bold",
            )

        ax00.set_xticks(range(n_models))
        rot = 26 if n_models > 6 else 0
        ha = "right" if rot > 0 else "center"
        ax00.set_xticklabels(df["Short_Label"], fontsize=8.2 if n_models > 8 else 9.5, rotation=rot, ha=ha)
        ax00.set_ylabel("Accuracy Disparity Gap", fontsize=11, fontweight="bold")
        ax00.set_title("A. Accuracy Gap Reduction (|Group 1 - Group 0|)", fontsize=12, fontweight="bold")

        y_gap_max = max(0.18, df["Accuracy_Gap"].max() * 1.25)
        ax00.set_ylim(0, y_gap_max)
        ax00.grid(axis="y", linestyle="--", alpha=0.5)

        if baseline_gap is not None:
            ax00.axhline(
                baseline_gap,
                color="#d62728",
                linestyle=":",
                linewidth=1.5,
                alpha=0.75,
                label=f"Biased Baseline Gap ({baseline_gap:.3f})",
            )
            ax00.legend(loc="upper right", frameon=True, fontsize=9.0)

        # ---------------------------------------------------------------------
        # Panel B: Subgroup Accuracies (Strictly Groups 0 and 1)
        # ---------------------------------------------------------------------
        ax01 = axes[0, 1]
        x = np.arange(n_models)
        w = max(0.26, min(0.38, 4.8 / n_models))

        # Check if sample counts are present for legend annotation
        leg0 = unprivileged_label
        leg1 = privileged_label
        if "N_Group_0" in df.columns and "N_Group_1" in df.columns:
            n0 = int(df["N_Group_0"].iloc[0])
            n1 = int(df["N_Group_1"].iloc[0])
            leg0 += f" (N={n0})"
            leg1 += f" (N={n1})"

        rects0 = ax01.bar(
            x - w / 2, df["Acc_Group_0"], w, label=leg0, color="#4575b4", edgecolor="black", alpha=0.9
        )
        rects1 = ax01.bar(
            x + w / 2, df["Acc_Group_1"], w, label=leg1, color="#d73027", edgecolor="black", alpha=0.9
        )

        for rect_set in [rects0, rects1]:
            for r in rect_set:
                h = r.get_height()
                font_sz = 8.0 if n_models <= 6 else (7.0 if n_models <= 10 else 6.2)
                ax01.text(
                    r.get_x() + r.get_width() / 2.0,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=font_sz,
                )

        ax01.set_xticks(x)
        ax01.set_xticklabels(df["Short_Label"], fontsize=8.2 if n_models > 8 else 9.5, rotation=rot, ha=ha)
        ax01.set_ylabel("Subgroup Accuracy", fontsize=11, fontweight="bold")
        ax01.set_title("B. Subgroup Accuracies (Privileged vs. Unprivileged)", fontsize=12, fontweight="bold")
        ax01.set_ylim(0, 1.15)
        ax01.legend(loc="lower left", frameon=True, fontsize=9.0)
        ax01.grid(axis="y", linestyle="--", alpha=0.5)

        # ---------------------------------------------------------------------
        # Panel C: Multidimensional Fairness Metrics
        # ---------------------------------------------------------------------
        ax10 = axes[1, 0]
        x_m = np.arange(n_models)
        w_m = max(0.10, min(0.19, 3.0 / n_models))

        eod_vals = np.abs(df.get("Equal_Opportunity_Diff", np.zeros(n_models)))
        spd_vals = np.abs(df.get("Statistical_Parity_Diff", np.zeros(n_models)))
        aod_vals = np.abs(df.get("Average_Odds_Diff", np.zeros(n_models)))
        c_disp_vals = df.get("C_Index_Disparity", np.zeros(n_models))

        ax10.bar(
            x_m - 1.5 * w_m,
            eod_vals,
            w_m,
            label="|Equal Opportunity Diff|",
            color="#2b83ba",
            edgecolor="black",
            alpha=0.85,
        )
        ax10.bar(
            x_m - 0.5 * w_m,
            spd_vals,
            w_m,
            label="|Statistical Parity Diff|",
            color="#abdda4",
            edgecolor="black",
            alpha=0.85,
        )
        ax10.bar(
            x_m + 0.5 * w_m,
            aod_vals,
            w_m,
            label="|Average Odds Diff|",
            color="#fdae61",
            edgecolor="black",
            alpha=0.85,
        )
        ax10.bar(
            x_m + 1.5 * w_m,
            c_disp_vals,
            w_m,
            label="C-Index Disparity",
            color="#d7191c",
            edgecolor="black",
            alpha=0.85,
        )

        ax10.set_xticks(x_m)
        ax10.set_xticklabels(df["Short_Label"], fontsize=8.2 if n_models > 8 else 9.5, rotation=rot, ha=ha)
        ax10.set_ylabel("Disparity Metric Magnitude", fontsize=11, fontweight="bold")
        ax10.set_title("C. Multidimensional Fairness Metrics", fontsize=12, fontweight="bold")

        max_c = max(0.35, float(c_disp_vals.max() * 1.25), float(spd_vals.max() * 1.2), float(eod_vals.max() * 1.2))
        ax10.set_ylim(0, min(1.0, max_c))
        ax10.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax10.grid(axis="y", linestyle="--", alpha=0.5)

        # ---------------------------------------------------------------------
        # Panel D: Performance vs. Fairness Trade-Off Frontier
        # ---------------------------------------------------------------------
        ax11 = axes[1, 1]

        gap_th, auc_th = ideal_thresholds
        ax11.axvline(gap_th, color="gray", linestyle="--", alpha=0.55)
        ax11.axhline(auc_th, color="gray", linestyle="--", alpha=0.55)

        # Place Ideal Region badge cleanly in top-left using transAxes
        ax11.text(
            0.03,
            0.96,
            "Ideal Region\n(High AUC, Low Gap)",
            transform=ax11.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="#1a9641",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#a3e4d7"),
        )

        ax11.set_xlabel("Accuracy Disparity Gap (Lower = Fairer)", fontsize=11, fontweight="bold")
        ax11.set_ylabel("Test AUROC (Higher = Better Discrimination)", fontsize=11, fontweight="bold")
        ax11.set_title("D. Performance vs. Fairness Trade-Off Frontier", fontsize=12, fontweight="bold")

        x_min = min(-0.015, float(df["Accuracy_Gap"].min()) - 0.03)
        x_max = max(0.20, float(df["Accuracy_Gap"].max()) + 0.08)
        y_min = min(0.40, float(df["AUC"].min()) - 0.06)
        y_max = max(0.85, float(df["AUC"].max()) + 0.06)

        ax11.set_xlim(x_min, x_max)
        ax11.set_ylim(y_min, y_max)
        ax11.grid(True, linestyle="--", alpha=0.5)

        # Plot Scatter Points
        for _, row in df.iterrows():
            ax11.scatter(
                row["Accuracy_Gap"],
                row["AUC"],
                color=row["Color"],
                s=190,
                edgecolor="black",
                linewidth=1.4,
                zorder=5,
                label=row.get("Model_Name", "Model"),
            )

        # Trace Pareto-Optimal Frontier curve
        px, py = self._compute_pareto_frontier(df["Accuracy_Gap"].values, df["AUC"].values)
        if len(px) > 1:
            ax11.plot(
                px, py, color="#2b83ba", linestyle="-.", linewidth=1.8, alpha=0.75, zorder=3, label="Pareto Frontier"
            )

        # Dynamic smart label placement with collision avoidance and subtle pointer arrows
        points_to_place = []
        for _, row in df.iterrows():
            clean_lbl = row["Short_Label"].replace("\n", " ")
            lbl_text = f"{clean_lbl}\n(AUC={row['AUC']:.3f}, Gap={row['Accuracy_Gap']:.3f})"
            points_to_place.append((float(row["Accuracy_Gap"]), float(row["AUC"]), lbl_text))

        placed = self._place_smart_labels(points_to_place, (x_min, x_max), (y_min, y_max))

        for x_pt, y_pt, tx, ty, lbl_text in placed:
            dist = np.hypot(tx - x_pt, ty - y_pt)
            arrowprops = None
            if dist > 0.018:
                arrowprops = dict(
                    arrowstyle="->",
                    lw=0.75,
                    color="#555555",
                    alpha=0.65,
                    shrinkA=3,
                    shrinkB=4,
                )

            ax11.annotate(
                lbl_text,
                (x_pt, y_pt),
                xytext=(tx, ty),
                ha="center",
                va="center",
                fontsize=7.8 if n_models > 8 else 8.5,
                fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
                arrowprops=arrowprops,
                zorder=6,
            )

        fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"[FairnessFigureGenerator] Saved 4-panel figure -> {save_path}")

        return fig, axes

