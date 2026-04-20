"""CNN planner model with a shared backbone and dynamic output heads."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..heads import HEAD_SPECS, HeadSpec
    from ..state_inputs import CURRENT_SPEED_KEY, StateInputConfig
except ImportError:
    from heads import HEAD_SPECS, HeadSpec
    from state_inputs import CURRENT_SPEED_KEY, StateInputConfig


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        activate: bool = True,
    ) -> None:
        super().__init__()
        resolved_padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=resolved_padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size=3, activate=False)
        if stride != 1 or in_channels != out_channels:
            self.projection = ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activate=False,
            )
        else:
            self.projection = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + residual)


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        block_count: int,
        downsample_first: bool,
    ) -> None:
        super().__init__()
        if block_count < 1:
            raise ValueError("block_count must be > 0")
        blocks: list[nn.Module] = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride=2 if downsample_first else 1,
            )
        ]
        for _ in range(1, block_count):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


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

        self.stem = ConvNormAct(
            in_channels=self.input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = ResidualStage(64, 64, block_count=2, downsample_first=False)
        self.stage2 = ResidualStage(64, 128, block_count=2, downsample_first=True)
        self.stage3 = ResidualStage(128, 256, block_count=2, downsample_first=True)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(in_features=256 * 2 * 2, out_features=256)
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
        x = self.stem(x)
        x = self.stem_pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

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

    @property
    def conv1(self) -> nn.Conv2d:
        # Preserve the legacy inspection path without registering the same module twice.
        return self.stem.conv
