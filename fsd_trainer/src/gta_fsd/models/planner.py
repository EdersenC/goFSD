"""CNN planner model with a shared backbone and dynamic output heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from heads import HEAD_SPECS, HeadSpec
from state_inputs import CURRENT_SPEED_KEY, StateInputConfig


class HeadBranch(nn.Module):
    def __init__(self, output_dim: int, *, in_features: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features=128, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.out(x)


class DrivingCNN(nn.Module):
    def __init__(
        self,
        frame_count: int = 3,
        head_specs: tuple[HeadSpec, ...] = HEAD_SPECS,
        *,
        state_input_config: StateInputConfig | None = None,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be > 0")
        self.frame_count = frame_count
        self.input_channels = frame_count * 3
        self.head_specs = head_specs
        self.state_input_config = state_input_config or StateInputConfig()

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(in_features=384, out_features=256)
        self.shared_relu = nn.ReLU()
        if self.state_input_config.current_speed_enabled:
            self.current_speed_encoder = nn.Sequential(
                nn.Linear(in_features=1, out_features=16),
                nn.ReLU(),
            )
        else:
            self.current_speed_encoder = None
        self.heads = nn.ModuleDict()
        for spec in self.head_specs:
            in_features = 256
            if spec.name in {"delta_speed", "future_speed"} and self.state_input_config.current_speed_enabled:
                in_features += 16
            self.heads[spec.name] = HeadBranch(spec.output_dim, in_features=in_features)

    def forward(self, x: torch.Tensor, current_speed: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        x = self.shared_relu(x)
        encoded_current_speed: torch.Tensor | None = None
        if self.state_input_config.current_speed_enabled:
            if current_speed is None:
                raise ValueError(f"{CURRENT_SPEED_KEY} is required for this checkpoint")
            speed = current_speed.float()
            if speed.ndim == 1:
                speed = speed.unsqueeze(-1)
            elif speed.ndim != 2 or speed.shape[-1] != 1:
                raise ValueError(
                    f"{CURRENT_SPEED_KEY} expected shape (batch,) or (batch, 1), got {tuple(speed.shape)}"
                )
            encoded_current_speed = self.current_speed_encoder(speed)

        outputs: dict[str, torch.Tensor] = {}
        for spec in self.head_specs:
            head_input = x
            if spec.name in {"delta_speed", "future_speed"} and encoded_current_speed is not None:
                head_input = torch.cat((x, encoded_current_speed), dim=-1)
            head_output = self.heads[spec.name](head_input)
            if spec.output_dim == 1:
                head_output = head_output.squeeze(-1)
            outputs[spec.name] = head_output
        return outputs
