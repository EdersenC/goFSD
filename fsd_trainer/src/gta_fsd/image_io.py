from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image, read_image


def _validate_rgb_tensor(image: Tensor, expected_size: tuple[int, int], *, source: str) -> Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Decoded image must have shape (3, H, W) for {source}, got {tuple(image.shape)}")

    expected_width, expected_height = expected_size
    _, actual_height, actual_width = image.shape
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"Frame image has unexpected size for {source}: "
            f"expected {expected_width}x{expected_height}, "
            f"found {actual_width}x{actual_height}"
        )
    return image


def _uint8_to_float_tensor(image: Tensor) -> Tensor:
    return image.to(dtype=torch.float32).div_(255.0)


def load_rgb_uint8_tensor_from_path(image_path: str | Path, expected_size: tuple[int, int]) -> Tensor:
    resolved_path = Path(image_path)
    image = read_image(str(resolved_path), mode=ImageReadMode.RGB)
    return _validate_rgb_tensor(image, expected_size, source=str(resolved_path))


def load_rgb_tensor_from_path(image_path: str | Path, expected_size: tuple[int, int]) -> Tensor:
    return _uint8_to_float_tensor(
        load_rgb_uint8_tensor_from_path(image_path, expected_size)
    )


def load_rgb_tensor_from_bytes(
    image_bytes: bytes,
    expected_size: tuple[int, int],
    *,
    source: str,
) -> Tensor:
    encoded = torch.tensor(bytearray(image_bytes), dtype=torch.uint8)
    image = decode_image(encoded, mode=ImageReadMode.RGB)
    return _uint8_to_float_tensor(
        _validate_rgb_tensor(image, expected_size, source=source)
    )
