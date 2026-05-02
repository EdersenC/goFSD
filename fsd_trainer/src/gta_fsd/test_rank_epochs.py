from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from rank_epochs import rank_epochs


class RankEpochTests(unittest.TestCase):
    def test_rank_epochs_prefers_lower_drive_score_over_lower_val_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "run_metrics.json"
            metrics_path.write_text(
                json.dumps({
                    "epochs": [
                        {
                            "epoch": 1,
                            "checkpoint": "epoch-001.pt",
                            "train_metrics": {"control_mae_overall": 0.10},
                            "val_metrics": {
                                "val_loss": 0.20,
                                "control_loss": 0.05,
                                "control_mae_overall": 0.10,
                                "drive_score": 0.10,
                            },
                        },
                        {
                            "epoch": 2,
                            "checkpoint": "epoch-002.pt",
                            "train_metrics": {"control_mae_overall": 0.10},
                            "val_metrics": {
                                "val_loss": 0.05,
                                "control_loss": 0.20,
                                "control_mae_overall": 0.30,
                                "drive_score": 0.30,
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )

            ranked = rank_epochs(metrics_path)

            self.assertEqual(ranked[0].epoch, 1)
            self.assertAlmostEqual(ranked[0].drive_score, 0.10, places=6)

    def test_rank_epochs_includes_future_speed_delta_loss_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "run_metrics.json"
            metrics_path.write_text(
                json.dumps({
                    "control_target_names": ["steering", "acceleration", "brakePressureAvg"],
                    "aux_target_names": ["future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"],
                    "epochs": [
                        {
                            "epoch": 1,
                            "checkpoint": "epoch-001.pt",
                            "train_metrics": {"control_mae_overall": 0.10},
                            "val_metrics": {
                                "val_loss": 0.20,
                                "control_loss": 0.06,
                                "control_mae_overall": 0.11,
                                "future_speed_delta_loss": 0.0123,
                                "future_speed_delta_mae": 0.0456,
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )

            ranked = rank_epochs(metrics_path)

            self.assertEqual(ranked[0].epoch, 1)
            self.assertIn("future_speed_delta_loss", ranked[0].target_loss_metrics)
            self.assertAlmostEqual(ranked[0].target_loss_metrics["future_speed_delta_loss"], 0.0123, places=6)
            self.assertIn("future_speed_delta", ranked[0].target_metrics)
            self.assertAlmostEqual(float(ranked[0].target_metrics["future_speed_delta"]["mae"]), 0.0456, places=6)

    def test_rank_epochs_handles_missing_optional_metrics_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "run_metrics.json"
            metrics_path.write_text(
                json.dumps({
                    "epochs": [
                        {
                            "epoch": 1,
                            "checkpoint": "epoch-001.pt",
                            "train_metrics": {},
                            "val_metrics": {
                                "loss": 0.30,
                            },
                        },
                    ],
                }),
                encoding="utf-8",
            )

            ranked = rank_epochs(metrics_path)

            self.assertEqual(len(ranked), 1)
            self.assertEqual(ranked[0].epoch, 1)
            self.assertGreater(len(ranked[0].missing_metrics), 0)
            self.assertTrue(math.isfinite(ranked[0].rank_score))


if __name__ == "__main__":
    unittest.main()
