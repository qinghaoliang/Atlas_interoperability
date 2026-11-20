"""Fully connected regressors shared by FLECHA, MECHA and the DNN baseline.

Layer order per hidden block follows the models in the first repository
release (commit 68ff5ba, FLECHA/model.py and MECHA/model_pytorch.py):
    Dropout -> Linear -> ReLU -> Norm
followed by Dropout -> Linear for the output layer. Linear weights use
Xavier-uniform initialization with zero bias; norm layers start at
weight 1, bias 0.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

NORMS = ("batch", "layer", "none")


@dataclass
class MLPSpec:
    input_size: int
    hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    output_size: int = 1
    dropout: float = 0.2
    norm: str = "batch"

    def as_dict(self) -> dict:
        return asdict(self)


class MLPRegressor(nn.Module):
    def __init__(self, spec: MLPSpec):
        super().__init__()
        if spec.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        layers: list[nn.Module] = []
        prev = spec.input_size
        for width in spec.hidden:
            layers += [nn.Dropout(spec.dropout), nn.Linear(prev, width), nn.ReLU()]
            if spec.norm == "batch":
                layers.append(nn.BatchNorm1d(width))
            elif spec.norm == "layer":
                layers.append(nn.LayerNorm(width))
            prev = width
        layers += [nn.Dropout(spec.dropout), nn.Linear(prev, spec.output_size)]
        self.net = nn.Sequential(*layers)
        self.spec = spec
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def nan_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over observed (non-NaN) target entries.

    Each observed entry has equal weight; with complete targets this equals the
    average of per-phenotype MSEs (Methods: "average mean squared error across
    the meta-training phenotypes").
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return (pred * 0).sum()
    diff = (pred - torch.where(mask, target, torch.zeros_like(target)))[mask]
    return (diff ** 2).mean()


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.spec.output_size))
