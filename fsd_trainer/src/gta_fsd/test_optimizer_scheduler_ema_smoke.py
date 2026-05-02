from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class OptimizerSchedulerEmaSmokeTests(unittest.TestCase):
    def test_one_epoch_smoke_with_adamw_cosine_and_ema(self) -> None:
        from train import (
            DEFAULT_LOSS_FUNCTION,
            DEFAULT_SMOOTH_L1_BETA,
            ModelEma,
            OptimizerConfig,
            SchedulerConfig,
            build_optimizer,
            build_scheduler,
            evaluate_epoch,
            train_epoch,
        )

        assert torch is not None
        assert nn is not None
        assert DataLoader is not None
        assert TensorDataset is not None

        torch.manual_seed(0)
        horizon = 2
        control_dim = 3
        aux_dim = 4
        sample_count = 8
        batch_size = 2

        images = torch.randn(sample_count, 5, 3, 16, 16)
        telemetry = torch.randn(sample_count, 9, 6)
        state_inputs = torch.randn(sample_count, 3)
        target_controls = torch.randn(sample_count, horizon, control_dim)
        target_aux = torch.randn(sample_count, horizon, aux_dim)

        loader = DataLoader(
            TensorDataset(images, telemetry, state_inputs, target_controls, target_aux),
            batch_size=batch_size,
            shuffle=False,
        )

        class TinyPlanner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(3, horizon * (control_dim + aux_dim))

            def forward(self, batch_images, batch_telemetry, batch_state_inputs):
                features = torch.stack(
                    (
                        batch_images.float().mean(dim=(1, 2, 3, 4)),
                        batch_telemetry.float().mean(dim=(1, 2)),
                        batch_state_inputs.float().mean(dim=1),
                    ),
                    dim=1,
                )
                logits = self.proj(features).view(-1, horizon, control_dim + aux_dim)
                return {
                    "pred_controls": logits[:, :, :control_dim],
                    "pred_aux": logits[:, :, control_dim:],
                }

        model = TinyPlanner()
        optimizer = build_optimizer(
            model,
            OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-4, grad_clip_norm=1.0),
            fallback_lr=1e-3,
        )
        scheduler = build_scheduler(
            optimizer,
            SchedulerConfig(name="cosine", warmup_fraction=0.05, min_lr_ratio=0.05, step_frequency="per_step"),
            total_train_steps=len(loader),
        )
        ema = ModelEma(model, decay=0.999)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        train_metrics, _, avg_timings = train_epoch(
            loader,
            optimizer,
            scheduler,
            ema,
            model,
            scaler,
            torch.device("cpu"),
            grad_clip_norm=1.0,
            future_offsets=(1, 2),
            state_input_names=(
                "route_direction_unknown",
                "route_direction_keep_straight",
                "route_direction_turn_left",
            ),
            control_target_names=("steering", "acceleration", "brakePressureAvg"),
            aux_target_names=("future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"),
            aux_loss_weight=0.4,
            horizon_loss_weights=(1.0, 1.0),
            target_loss_weights={},
            loss_function=DEFAULT_LOSS_FUNCTION,
            smooth_l1_beta=DEFAULT_SMOOTH_L1_BETA,
            log_every_n_batches=100,
            target_transforms=None,
        )

        ema.apply_to(model)
        try:
            val_metrics, _ = evaluate_epoch(
                loader,
                model,
                torch.device("cpu"),
                future_offsets=(1, 2),
                control_target_names=("steering", "acceleration", "brakePressureAvg"),
                aux_target_names=("future_speed", "future_speed_delta", "future_yaw_delta", "future_yaw_rate"),
                target_transforms=None,
                aux_loss_weight=0.4,
                horizon_loss_weights=(1.0, 1.0),
                target_loss_weights={},
                loss_function=DEFAULT_LOSS_FUNCTION,
                smooth_l1_beta=DEFAULT_SMOOTH_L1_BETA,
            )
        finally:
            ema.restore(model)

        self.assertGreaterEqual(float(train_metrics["loss"]), 0.0)
        self.assertGreaterEqual(float(val_metrics["val_loss"]), 0.0)
        self.assertIsNotNone(avg_timings["grad_norm"])


if __name__ == "__main__":
    unittest.main()
