from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config import STOP_SIGN_AUX_TARGET_NAMES
from control_contract import STOP_SIGN_CONTROL_TARGET_NAMES
from rank_epochs import rank_epochs


def epoch_metrics(epoch: int, *, score: float, val_loss: float) -> dict[str, object]:
    val_metrics: dict[str, float] = {
        "val_loss": val_loss,
        "control_loss": 0.2,
        "control_mae_overall": 0.1,
        "control_mae_overall_denorm": 0.25,
        "aux_mae_overall": 0.2,
        "aux_mae_overall_denorm": 0.2,
        "stop_sign_score": score,
    }
    for name in (*STOP_SIGN_CONTROL_TARGET_NAMES, *STOP_SIGN_AUX_TARGET_NAMES):
        val_metrics[f"{name}_loss"] = 0.01 * epoch
        val_metrics[f"{name}_mae"] = 0.02 * epoch
    return {
        "epoch": epoch,
        "checkpoint": f"epoch-{epoch:03d}.pt",
        "train_metrics": {"control_mae_overall": 0.05},
        "val_metrics": val_metrics,
    }


class RankEpochTests(unittest.TestCase):
    def test_rank_epochs_prefers_lower_stop_sign_score_over_val_loss(self) -> None:
        payload = {
            "control_target_names": list(STOP_SIGN_CONTROL_TARGET_NAMES),
            "aux_target_names": list(STOP_SIGN_AUX_TARGET_NAMES),
            "epochs": [
                epoch_metrics(1, score=0.10, val_loss=0.50),
                epoch_metrics(2, score=0.30, val_loss=0.05),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            ranked = rank_epochs(path)
        self.assertEqual(ranked[0].epoch, 1)
        self.assertAlmostEqual(ranked[0].stop_sign_score, 0.10)

    def test_rank_epochs_reports_motion_plan_and_diagnostic_targets(self) -> None:
        payload = {
            "control_target_names": list(STOP_SIGN_CONTROL_TARGET_NAMES),
            "aux_target_names": list(STOP_SIGN_AUX_TARGET_NAMES),
            "epochs": [epoch_metrics(1, score=0.10, val_loss=0.20)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            ranked = rank_epochs(path)
        self.assertIn("future_speed_mps_loss", ranked[0].target_loss_metrics)
        self.assertIn("actual_brake_pressure", ranked[0].target_metrics)

    def test_rank_epochs_rejects_non_stop_sign_target_contract(self) -> None:
        payload = {
            "control_target_names": ["steering", "acceleration"],
            "aux_target_names": list(STOP_SIGN_AUX_TARGET_NAMES),
            "epochs": [epoch_metrics(1, score=0.10, val_loss=0.20)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stop-sign motion-plan targets"):
                rank_epochs(path)


if __name__ == "__main__":
    unittest.main()
