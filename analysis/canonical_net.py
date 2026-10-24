"""Canonical LitePI model metadata, preprocessing, and checkpoint loading.

The three canonical networks are the ones trained by ``analysis/train_hybrid.py``:

    MNIST  : Conv(4,7x7,s3) -> act -> FC(256->64)  -> act -> FC(64->10)
    CIFAR  : Conv(12,5x5,s3) -> act -> FC(1200->128) -> act -> FC(128->10)
    UCI HAR: Conv(12,3x9,s2) -> act -> FC(756->128) -> act -> FC(128->6)

This module is shared by the protocol server and by mask-domain export so the
wire geometry, client preprocessing, and training architecture cannot drift
apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from params_pi import GLOBAL_SCALE

A0, A1, A2 = 0.0625, 0.5, 0.5

CIFAR_GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)
CIFAR_GRAY_MEAN = (
    0.2989 * 0.4914 + 0.5870 * 0.4822 + 0.1140 * 0.4465
)
CIFAR_GRAY_STD = (
    0.2989 * 0.2470 + 0.5870 * 0.2435 + 0.1140 * 0.2616
)


@dataclass(frozen=True)
class CanonicalSpec:
    name: str
    raw_shape: tuple[int, ...]
    model_shape: tuple[int, int, int]
    im2col_kernel: tuple[int, int]
    stride: tuple[int, int]
    windows: int
    encoded_slots: int
    conv_channels: int
    conv_flat: int
    hidden: int
    classes: int
    parameters: int

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.raw_shape))


CANONICAL_SPECS: dict[str, CanonicalSpec] = {
    "mnist": CanonicalSpec(
        name="mnist",
        raw_shape=(1, 28, 28),
        model_shape=(1, 28, 28),
        im2col_kernel=(7, 7),
        stride=(3, 3),
        windows=64,
        encoded_slots=4096,
        conv_channels=4,
        conv_flat=256,
        hidden=64,
        classes=10,
        parameters=17_298,
    ),
    "cifar10": CanonicalSpec(
        name="cifar10",
        raw_shape=(3, 32, 32),
        model_shape=(1, 32, 32),
        im2col_kernel=(5, 5),
        stride=(3, 3),
        windows=100,
        encoded_slots=3200,
        conv_channels=12,
        conv_flat=1200,
        hidden=128,
        classes=10,
        parameters=155_330,
    ),
    "uci_har": CanonicalSpec(
        name="uci_har",
        raw_shape=(128, 9),
        model_shape=(1, 128, 9),
        im2col_kernel=(3, 9),
        stride=(2, 1),
        windows=63,
        encoded_slots=2016,
        conv_channels=12,
        conv_flat=756,
        hidden=128,
        classes=6,
        parameters=98_006,
    ),
}


def choose_checkpoint(root: Path, dataset: str, schedule: tuple[int, ...],
                      seed: int, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    tag = "".join(str(bit) for bit in schedule)
    preferred = root / "datas" / f"{dataset}_hybrid_s{tag}_seed{seed}_cnn_train.pt"
    if preferred.exists():
        return preferred
    fallback = root / "datas" / f"{dataset}_hybrid_s{tag}_cnn_train.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"canonical checkpoint not found: {preferred} (or {fallback.name})"
    )


class CanonicalCNN:
    """Canonical network with the exact per-layer P2/ReLU schedule."""

    architecture = "canonical_cnn"

    def __init__(self, dataset: str, schedule: tuple[int, ...], seed: int,
                 checkpoint: Path, root: Path) -> None:
        import torch

        if dataset not in CANONICAL_SPECS:
            raise ValueError(f"unknown canonical dataset {dataset!r}")
        if len(schedule) != 2 or any(bit not in (0, 1) for bit in schedule):
            raise ValueError("canonical models require a two-bit schedule")
        self.spec = CANONICAL_SPECS[dataset]
        self.dataset = dataset
        self.schedule = tuple(int(bit) for bit in schedule)
        self.seed = int(seed)
        self.root = Path(root)
        self.checkpoint = Path(checkpoint)
        self.input_dim = self.spec.input_dim
        self.input_slots = self.spec.encoded_slots
        self.windows = self.spec.windows
        self.dims = [
            self.input_dim,
            self.spec.conv_flat,
            self.spec.hidden,
            self.spec.classes,
        ]
        self.channels = self.spec.conv_channels
        self.stride = self.spec.stride
        self.im2col_kernel = self.spec.im2col_kernel

        state = self._load_state(self.checkpoint)
        self.conv_weight = state["conv1.weight"].detach().cpu().to(torch.float64)
        self.conv_bias = state["conv1.bias"].detach().cpu().to(torch.float64)
        self.fc1_weight = state["fc1.weight"].detach().cpu().to(torch.float64)
        self.fc1_bias = state["fc1.bias"].detach().cpu().to(torch.float64)
        self.fc2_weight = state["fc2.weight"].detach().cpu().to(torch.float64)
        self.fc2_bias = state["fc2.bias"].detach().cpu().to(torch.float64)
        self.validate_weights()
        # Offline server-side weight cache used by LinearEval.
        self.conv_weight_np = self.conv_weight.numpy()
        self.conv_bias_np = self.conv_bias.numpy()
        self.fc1_weight_t_np = self.fc1_weight.T.numpy()
        self.fc1_bias_np = self.fc1_bias.numpy()
        self.fc2_weight_t_np = self.fc2_weight.T.numpy()
        self.fc2_bias_np = self.fc2_bias.numpy()

        self.preprocess_meta = self._preprocess_meta()

    @staticmethod
    def _load_state(path: Path) -> dict[str, torch.Tensor]:
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise ValueError(f"checkpoint {path} does not contain a state_dict")
        required = {
            "conv1.weight", "conv1.bias",
            "fc1.weight", "fc1.bias",
            "fc2.weight", "fc2.bias",
        }
        missing = required.difference(obj)
        if missing:
            raise ValueError(f"checkpoint {path} is missing {sorted(missing)}")
        return obj

    def validate_weights(self) -> None:
        import torch

        s = self.spec
        expected = {
            "conv1.weight": (s.conv_channels, 1, *self._torch_kernel()),
            "conv1.bias": (s.conv_channels,),
            "fc1.weight": (s.hidden, s.conv_flat),
            "fc1.bias": (s.hidden,),
            "fc2.weight": (s.classes, s.hidden),
            "fc2.bias": (s.classes,),
        }
        actual = {
            "conv1.weight": tuple(self.conv_weight.shape),
            "conv1.bias": tuple(self.conv_bias.shape),
            "fc1.weight": tuple(self.fc1_weight.shape),
            "fc1.bias": tuple(self.fc1_bias.shape),
            "fc2.weight": tuple(self.fc2_weight.shape),
            "fc2.bias": tuple(self.fc2_bias.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(
                    f"{name} shape {actual[name]} does not match canonical {shape}"
                )
        for name in expected:
            tensor = getattr(self, name.replace(".", "_"))
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN or infinity")

    def _torch_kernel(self) -> tuple[int, int]:
        if self.spec.name == "uci_har":
            return (3, 9)
        return self.spec.im2col_kernel

    def _preprocess_meta(self) -> dict[str, Any]:
        if self.dataset == "cifar10":
            return {
                "mode": "cifar10_gray_zscore",
                "raw_shape": list(self.spec.raw_shape),
                "model_shape": list(self.spec.model_shape),
                "gray_weights": list(CIFAR_GRAY_WEIGHTS),
                "mean": CIFAR_GRAY_MEAN,
                "std": CIFAR_GRAY_STD,
            }
        if self.dataset == "uci_har":
            stats = self.root / "datas" / "uci_har_he_stats.npz"
            with np.load(stats) as cached:
                mean = np.asarray(cached["mean"], dtype=np.float64)
                std = np.asarray(cached["std"], dtype=np.float64)
            return {
                "mode": "per_channel_zscore",
                "raw_shape": list(self.spec.raw_shape),
                "model_shape": list(self.spec.model_shape),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        return {
            "mode": "identity",
            "raw_shape": list(self.spec.raw_shape),
            "model_shape": list(self.spec.model_shape),
        }

    def client_spec(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "dataset": self.dataset,
            "dims": self.dims,
            "input_dim": self.input_dim,
            "input_slots": self.input_slots,
            "windows": self.windows,
            "im2col_kernel": list(self.im2col_kernel),
            "stride": list(self.stride),
            "encoding_stride": 2 if self.dataset == "uci_har" else self.stride[0],
            "nonlinear_dims": [self.spec.conv_flat, self.spec.hidden],
            "preprocess": self.preprocess_meta,
            "schedule": list(self.schedule),
            "hybrid_layers": self.hybrid_layers,
            "encoding": "im2col",
        }

    @property
    def hybrid_layers(self) -> int:
        return sum(self.schedule)

    def _model_input(self, raw: np.ndarray) -> torch.Tensor:
        import torch

        arr = np.asarray(raw, dtype=np.float64).ravel()
        if arr.size != self.input_dim:
            raise ValueError(f"input has {arr.size} values, model wants {self.input_dim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("input contains NaN or infinity")
        shaped = arr.reshape(self.spec.raw_shape)
        meta = self.preprocess_meta
        if self.dataset == "cifar10":
            r, g, b = shaped[0], shaped[1], shaped[2]
            gray = (
                CIFAR_GRAY_WEIGHTS[0] * r
                + CIFAR_GRAY_WEIGHTS[1] * g
                + CIFAR_GRAY_WEIGHTS[2] * b
            )
            shaped = (gray - CIFAR_GRAY_MEAN) / CIFAR_GRAY_STD
            shaped = shaped.reshape(self.spec.model_shape)
        elif self.dataset == "uci_har":
            mean = np.asarray(meta["mean"], dtype=np.float64)
            std = np.asarray(meta["std"], dtype=np.float64)
            shaped = (shaped - mean) / std
            shaped = shaped.reshape(self.spec.model_shape)
        else:
            shaped = shaped.reshape(self.spec.model_shape)
        return torch.from_numpy(np.asarray(shaped, dtype=np.float64)).unsqueeze(0)

    def prepare_array(self, raw: np.ndarray) -> np.ndarray:
        return self._model_input(raw).squeeze(0).numpy()

    def plain_forward(self, raw: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.plain_forward_prepared(self._model_input(raw))

    def plain_forward_prepared(self, prepared) -> tuple[np.ndarray, list[np.ndarray]]:
        import torch
        import torch.nn.functional as F

        if isinstance(prepared, np.ndarray):
            x = torch.from_numpy(np.asarray(prepared, dtype=np.float64))
        else:
            x = prepared.detach().cpu().to(torch.float64)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if tuple(x.shape[1:]) != self.spec.model_shape:
            raise ValueError(
                f"prepared input shape {tuple(x.shape)} does not match "
                f"{self.spec.model_shape}"
            )
        z1 = F.conv2d(
            x, self.conv_weight, self.conv_bias,
            stride=self.spec.stride,
        )
        a1 = F.relu(z1) if self.schedule[0] else A0 + A1 * z1 + A2 * z1 * z1
        z2 = F.linear(a1.flatten(1), self.fc1_weight, self.fc1_bias)
        a2 = F.relu(z2) if self.schedule[1] else A0 + A1 * z2 + A2 * z2 * z2
        logits = F.linear(a2, self.fc2_weight, self.fc2_bias)
        spatial = int(np.prod(tuple(z1.shape[2:])))
        if z1.shape[1] != self.channels or spatial != self.windows:
            raise RuntimeError(f"conv output geometry is {tuple(z1.shape)}")
        if z2.ndim != 2 or z2.shape[1] != self.spec.hidden:
            raise RuntimeError(f"FC1 output geometry is {tuple(z2.shape)}")
        return logits.flatten().numpy(), [z1.flatten().numpy(), z2.flatten().numpy()]

    def mask_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, bit in enumerate(self.schedule) if bit)

    def output_layer_id(self) -> int:
        return len(self.schedule)


def load_canonical_model(root: Path, dataset: str, schedule: tuple[int, ...],
                         seed: int, checkpoint: Path | None = None) -> CanonicalCNN:
    path = choose_checkpoint(root, dataset, schedule, seed, checkpoint)
    return CanonicalCNN(dataset, schedule, seed, path, root)


def quantization_step() -> float:
    return 1.0 / float(GLOBAL_SCALE)
