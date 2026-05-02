"""Temporal planner model with a CNN vision path and GRU telemetry path."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_width(base: int, width_multiplier: float) -> int:
    return max(8, int(math.ceil((base * width_multiplier) / 8.0) * 8))


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
        return self.relu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
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
        return self.relu(self.conv2(self.conv1(x)) + residual)


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
            ResidualBlock(in_channels, out_channels, stride=2 if downsample_first else 1)
        ]
        for _ in range(1, block_count):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def build_horizon_decoder(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DrivingCNN(nn.Module):
    def __init__(
        self,
        frame_count: int = 5,
        *,
        telemetry_feature_dim: int = 6,
        telemetry_hidden_dim: int = 128,
        telemetry_sequence_length: int = 9,
        horizon: int = 6,
        control_dim: int = 2,
        state_input_dim: int = 0,
        aux_dim: int = 4,
        width_multiplier: float = 1.0,
        dropout: float = 0.1,
        visual_temporal_enabled: bool = True,
        visual_temporal_type: str = "gru",
        visual_temporal_hidden_dim: int = 256,
        visual_temporal_num_layers: int = 1,
        visual_temporal_bidirectional: bool = False,
        visual_temporal_dropout: float = 0.0,
        horizon_decoder_enabled: bool = True,
        horizon_embed_dim: int = 32,
        horizon_decoder_hidden_dim: int = 256,
        horizon_decoder_num_layers: int = 2,
        horizon_decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be > 0")
        if telemetry_feature_dim < 1:
            raise ValueError("telemetry_feature_dim must be > 0")
        if telemetry_hidden_dim < 1:
            raise ValueError("telemetry_hidden_dim must be > 0")
        if telemetry_sequence_length < 1:
            raise ValueError("telemetry_sequence_length must be > 0")
        if horizon < 1:
            raise ValueError("horizon must be > 0")
        if control_dim < 1:
            raise ValueError("control_dim must be > 0")
        if aux_dim < 1:
            raise ValueError("aux_dim must be > 0")
        if state_input_dim < 0:
            raise ValueError("state_input_dim must be >= 0")
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be > 0")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0]")
        if visual_temporal_type.strip().lower() != "gru":
            raise ValueError("visual_temporal_type must be 'gru'")
        if visual_temporal_hidden_dim < 1:
            raise ValueError("visual_temporal_hidden_dim must be > 0")
        if visual_temporal_num_layers < 1:
            raise ValueError("visual_temporal_num_layers must be > 0")
        if not 0.0 <= visual_temporal_dropout <= 1.0:
            raise ValueError("visual_temporal_dropout must be in [0.0, 1.0]")
        if not horizon_decoder_enabled:
            raise ValueError("horizon decoder is required for the temporal planner")
        if horizon_embed_dim < 1:
            raise ValueError("horizon_embed_dim must be > 0")
        if horizon_decoder_hidden_dim < 1:
            raise ValueError("horizon_decoder_hidden_dim must be > 0")
        if horizon_decoder_num_layers < 1:
            raise ValueError("horizon_decoder_num_layers must be > 0")
        if not 0.0 <= horizon_decoder_dropout <= 1.0:
            raise ValueError("horizon_decoder_dropout must be in [0.0, 1.0]")

        self.frame_count = frame_count
        self.telemetry_feature_dim = telemetry_feature_dim
        self.telemetry_hidden_dim = telemetry_hidden_dim
        self.telemetry_sequence_length = telemetry_sequence_length
        self.horizon = horizon
        self.control_dim = control_dim
        self.state_input_dim = state_input_dim
        self.aux_dim = aux_dim
        self.input_channels = 3
        self.dropout = float(dropout)
        self.visual_temporal_enabled = bool(visual_temporal_enabled)
        self.visual_temporal_type = visual_temporal_type.strip().lower()
        self.visual_temporal_hidden_dim = visual_temporal_hidden_dim
        self.visual_temporal_num_layers = visual_temporal_num_layers
        self.visual_temporal_bidirectional = bool(visual_temporal_bidirectional)
        self.visual_temporal_dropout = float(visual_temporal_dropout)
        self.horizon_decoder_enabled = bool(horizon_decoder_enabled)
        self.horizon_embed_dim = horizon_embed_dim
        self.horizon_decoder_hidden_dim = horizon_decoder_hidden_dim
        self.horizon_decoder_num_layers = horizon_decoder_num_layers
        self.horizon_decoder_dropout = float(horizon_decoder_dropout)

        stem_width = scaled_width(64, width_multiplier)
        stage2_width = scaled_width(128, width_multiplier)
        stage3_width = scaled_width(256, width_multiplier)
        fusion_width = scaled_width(256, width_multiplier)
        head_hidden_width = scaled_width(128, width_multiplier)

        self.stem = ConvNormAct(
            in_channels=3,
            out_channels=stem_width,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = ResidualStage(stem_width, stem_width, block_count=2, downsample_first=False)
        self.stage2 = ResidualStage(stem_width, stage2_width, block_count=2, downsample_first=True)
        self.stage3 = ResidualStage(stage2_width, stage3_width, block_count=2, downsample_first=True)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.vision_fc = nn.Sequential(
            nn.Linear(stage3_width * 2 * 2, fusion_width),
            nn.ReLU(),
            nn.LayerNorm(fusion_width),
        )
        visual_temporal_width = (
            visual_temporal_hidden_dim * (2 if self.visual_temporal_bidirectional else 1)
            if self.visual_temporal_enabled
            else fusion_width
        )
        self.visual_temporal_gru = (
            nn.GRU(
                input_size=fusion_width,
                hidden_size=visual_temporal_hidden_dim,
                num_layers=visual_temporal_num_layers,
                batch_first=True,
                bidirectional=self.visual_temporal_bidirectional,
                dropout=visual_temporal_dropout if visual_temporal_num_layers > 1 else 0.0,
            )
            if self.visual_temporal_enabled
            else nn.Identity()
        )
        self.visual_temporal_norm = nn.LayerNorm(visual_temporal_width)

        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_feature_dim, telemetry_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(telemetry_hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.telemetry_gru = nn.GRU(
            input_size=telemetry_hidden_dim,
            hidden_size=telemetry_hidden_dim,
            batch_first=True,
        )
        self.telemetry_norm = nn.LayerNorm(telemetry_hidden_dim)
        self.state_encoder = (
            nn.Sequential(
                nn.Linear(state_input_dim, state_input_dim),
                nn.ReLU(),
                nn.LayerNorm(state_input_dim),
                nn.Dropout(self.dropout),
            )
            if state_input_dim > 0
            else nn.Identity()
        )
        self.fusion = nn.Sequential(
            nn.Linear(visual_temporal_width + telemetry_hidden_dim + state_input_dim, fusion_width),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_width, head_hidden_width),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.context_dim = head_hidden_width
        self.horizon_embeddings = nn.Embedding(horizon, horizon_embed_dim)
        decoder_input_dim = self.context_dim + horizon_embed_dim
        self.control_decoder = build_horizon_decoder(
            decoder_input_dim,
            horizon_decoder_hidden_dim,
            self.control_dim,
            num_layers=horizon_decoder_num_layers,
            dropout=horizon_decoder_dropout,
        )
        self.aux_decoder = build_horizon_decoder(
            decoder_input_dim,
            horizon_decoder_hidden_dim,
            self.aux_dim,
            num_layers=horizon_decoder_num_layers,
            dropout=horizon_decoder_dropout,
        )

    def _encode_frame_features(self, frame_images: torch.Tensor) -> torch.Tensor:
        if frame_images.device.type == "cuda":
            frame_images = frame_images.contiguous(memory_format=torch.channels_last)
        features = self.stem_pool(self.stem(frame_images))
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        return self.vision_fc(self.flatten(self.pool(features)))

    def _encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = images.shape
        if torch.is_floating_point(images):
            normalized_images = images.float()
        else:
            normalized_images = images.to(dtype=torch.float32).div_(255.0)

        # The dataset provides frames in chronological order:
        # images[:, 0] is the oldest frame and images[:, -1] is newest/current.
        frame_images = normalized_images.reshape(batch_size * time_steps, channels, height, width)
        frame_embeddings = self._encode_frame_features(frame_images)
        visual_sequence = frame_embeddings.reshape(batch_size, time_steps, -1)
        if self.visual_temporal_enabled:
            _, hidden = self.visual_temporal_gru(visual_sequence)
            if self.visual_temporal_bidirectional:
                visual_vec = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            else:
                visual_vec = hidden[-1]
        else:
            visual_vec = visual_sequence[:, -1]
        return self.visual_temporal_norm(visual_vec)

    def forward(
        self,
        images: torch.Tensor,
        telemetry: torch.Tensor,
        state_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"images must have rank 5 [B, T_img, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != self.frame_count:
            raise ValueError(
                f"images expected T_img={self.frame_count}, got T_img={int(images.shape[1])}"
            )
        if images.shape[2] != 3:
            raise ValueError(f"images expected 3 channels per frame, got {int(images.shape[2])}")
        if telemetry.ndim != 3:
            raise ValueError(f"telemetry must have rank 3 [B, T_tel, F], got {tuple(telemetry.shape)}")
        if telemetry.shape[1] != self.telemetry_sequence_length:
            raise ValueError(
                f"telemetry expected T_tel={self.telemetry_sequence_length}, got T_tel={int(telemetry.shape[1])}"
            )
        if telemetry.shape[2] != self.telemetry_feature_dim:
            raise ValueError(
                "telemetry feature count mismatch: "
                f"expected {self.telemetry_feature_dim}, got {int(telemetry.shape[2])}"
            )
        if images.shape[0] != telemetry.shape[0]:
            raise ValueError(
                "images and telemetry batch sizes must match: "
                f"images={int(images.shape[0])} telemetry={int(telemetry.shape[0])}"
            )
        if self.state_input_dim > 0:
            if state_inputs is None:
                raise ValueError(f"state_inputs must be provided with shape [B, {self.state_input_dim}]")
            if state_inputs.ndim != 2:
                raise ValueError(
                    f"state_inputs must have rank 2 [B, S], got {tuple(state_inputs.shape)}"
                )
            if state_inputs.shape[0] != images.shape[0]:
                raise ValueError(
                    "state_inputs batch size must match images batch size: "
                    f"state_inputs={int(state_inputs.shape[0])} images={int(images.shape[0])}"
                )
            if state_inputs.shape[1] != self.state_input_dim:
                raise ValueError(
                    "state_inputs width mismatch: "
                    f"expected {self.state_input_dim}, got {int(state_inputs.shape[1])}"
                )
        elif state_inputs is not None and state_inputs.numel() > 0:
            raise ValueError(
                f"state_inputs were provided but this model was built with state_input_dim={self.state_input_dim}"
            )

        vision_vec = self._encode_vision(images)
        _, hidden = self.telemetry_gru(self.telemetry_encoder(telemetry.float()))
        telemetry_vec = self.telemetry_norm(hidden[-1])
        fused_parts = [vision_vec, telemetry_vec]
        if self.state_input_dim > 0 and state_inputs is not None:
            fused_parts.append(self.state_encoder(state_inputs.float()))
        context = self.fusion(torch.cat(fused_parts, dim=-1))

        horizon_tokens = self.horizon_embeddings.weight.unsqueeze(0).expand(context.shape[0], -1, -1)
        context_expanded = context.unsqueeze(1).expand(-1, self.horizon, -1)
        decoder_input = torch.cat((context_expanded, horizon_tokens), dim=-1)
        pred_controls = self.control_decoder(decoder_input)
        pred_aux = self.aux_decoder(decoder_input)
        if pred_controls.ndim != 3 or pred_controls.shape[1:] != (self.horizon, self.control_dim):
            raise ValueError(
                "control_decoder shape check failed: "
                f"expected [B, {self.horizon}, {self.control_dim}], got {tuple(pred_controls.shape)}"
            )
        if pred_aux.ndim != 3 or pred_aux.shape[1:] != (self.horizon, self.aux_dim):
            raise ValueError(
                "aux_decoder shape check failed: "
                f"expected [B, {self.horizon}, {self.aux_dim}], got {tuple(pred_aux.shape)}"
            )
        return {
            "pred_controls": pred_controls,
            "pred_aux": pred_aux,
        }

    @property
    def conv1(self) -> nn.Conv2d:
        return self.stem.conv
