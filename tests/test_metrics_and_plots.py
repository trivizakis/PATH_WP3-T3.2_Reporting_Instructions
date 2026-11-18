# -*- coding: utf-8 -*-
"""
Unit tests for path_fairness toolkit.
"""

import os
import unittest
import numpy as np
from path_fairness import (
    FairnessMetricsReporter,
    FairnessFigureGenerator,
    generate_synthetic_fairness_dataset,
)


class TestPathFairness(unittest.TestCase):

    def setUp(self):
        self.data = generate_synthetic_fairness_dataset(
            n_samples=200, n_features=6, random_state=42
        )

    def test_dataset_generation(self):
        self.assertEqual(len(self.data["y_train"]) + len(self.data["y_test"]), 200)
        self.assertTrue(set(np.unique(self.data["s_train"])).issubset({0, 1}))
        self.assertTrue(set(np.unique(self.data["y_train"])).issubset({0, 1}))

    def test_metrics_reporter(self):
        reporter = FairnessMetricsReporter()
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.7, 0.6])
        s_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        rec = reporter.add_evaluation(
            model_name="TestModel",
            variant="TestVariant",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            s_true=s_true,
        )

        self.assertIn("AUC", rec)
        self.assertIn("Accuracy_Gap", rec)
        self.assertIn("Equal_Opportunity_Diff", rec)
        self.assertIn("C_Index_Disparity", rec)
        self.assertGreaterEqual(rec["Accuracy"], 0.0)
        self.assertLessEqual(rec["Accuracy"], 1.0)

        df = reporter.get_summary_dataframe()
        self.assertEqual(len(df), 1)

    def test_figure_generation(self):
        reporter = FairnessMetricsReporter()
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        s_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Model A
        reporter.add_evaluation(
            model_name="ModelA",
            variant="Biased",
            y_true=y_true,
            y_pred=np.array([0, 1, 0, 0, 0, 1, 1, 1]),
            y_prob=np.array([0.1, 0.8, 0.2, 0.4, 0.3, 0.9, 0.7, 0.6]),
            s_true=s_true,
        )
        # Model B
        reporter.add_evaluation(
            model_name="ModelB",
            variant="Reweighing",
            y_true=y_true,
            y_pred=np.array([0, 1, 0, 1, 0, 1, 0, 0]),
            y_prob=np.array([0.2, 0.8, 0.1, 0.7, 0.3, 0.9, 0.4, 0.2]),
            s_true=s_true,
        )

        df = reporter.get_summary_dataframe()
        gen = FairnessFigureGenerator()
        out_png = "/tmp/test_fairness_fig.png"
        fig, axes = gen.generate_4panel_figure(
            df_metrics=df,
            save_path=out_png,
            dpi=100,
        )
        self.assertTrue(os.path.exists(out_png))
        if os.path.exists(out_png):
            os.remove(out_png)


if __name__ == "__main__":
    unittest.main()

