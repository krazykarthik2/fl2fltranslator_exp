"""Multi-task auxiliary classification heads for ownership, mutability, lifetime, unsafe."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class OwnershipClassifier(nn.Module):
    """Predict: owned(0), borrowed(1), borrowed_mut(2), raw_ptr(3)."""

    LABELS = ["owned", "borrowed", "borrowed_mut", "raw_ptr"]

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 4)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.head(encoder_output)  # (B, T, 4)


class MutabilityClassifier(nn.Module):
    """Predict: immutable(0), mutable(1)."""

    LABELS = ["immutable", "mutable"]

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 2)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.head(encoder_output)  # (B, T, 2)


class LifetimeClassifier(nn.Module):
    """Predict: static(0), local(1), parameter(2), heap(3)."""

    LABELS = ["static", "local", "parameter", "heap"]

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 4)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.head(encoder_output)  # (B, T, 4)


class UnsafeClassifier(nn.Module):
    """Predict: safe(0), unsafe(1)."""

    LABELS = ["safe", "unsafe"]

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 2)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.head(encoder_output)  # (B, T, 2)


class MultiTaskHead(nn.Module):
    """Combines all four auxiliary classification heads."""

    def __init__(self, d_model: int):
        super().__init__()
        self.ownership = OwnershipClassifier(d_model)
        self.mutability = MutabilityClassifier(d_model)
        self.lifetime = LifetimeClassifier(d_model)
        self.unsafe_head = UnsafeClassifier(d_model)

    def forward(self, encoder_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "ownership":  self.ownership(encoder_output),
            "mutability": self.mutability(encoder_output),
            "lifetime":   self.lifetime(encoder_output),
            "unsafe":     self.unsafe_head(encoder_output),
        }

    def __repr__(self) -> str:
        return "MultiTaskHead(ownership=4, mutability=2, lifetime=4, unsafe=2)"
