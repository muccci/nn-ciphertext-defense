from __future__ import annotations

import sys
import torch
import torch.nn as nn

from attack.common.cases import CaseSpec, effective_trace_len
from attack.common.util import COMMON_DIR


def _ensure_lpips_path() -> None:
    root = str(COMMON_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


class TraceTransform(nn.Module):
    def __init__(self, cipher, spec: CaseSpec):
        super().__init__()
        if spec.fold is None:
            self.enc = cipher.dense_trace_encoder(effective_trace_len(spec), spec.nz, spec.nz)
        else:
            ctor = cipher.__dict__[f"trace_encoder_{spec.fold[2]}"]
            self.enc = ctor(dim=spec.nz, nc=spec.fold[0])
        self.dec = cipher.__dict__[f"image_decoder_{spec.height}"](dim=spec.nz, nc=spec.nc)
        self.dense = spec.fold is None

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        if self.dense:
            trace = trace.view(trace.shape[0], -1)
        return self.dec(self.enc(trace))


class ImageAE(nn.Module):
    def __init__(self, cipher, spec: CaseSpec):
        super().__init__()
        self.enc = cipher.__dict__[f"image_encoder_{spec.height}"](dim=spec.nz, nc=spec.nc)
        self.dec = cipher.__dict__[f"image_decoder_{spec.height}"](dim=spec.nz, nc=spec.nc)

    def forward(self, image_model: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(image_model))


class ReconstructionLoss(nn.Module):
    def __init__(self, name: str, device: torch.device, lpips_net: str = "alex"):
        super().__init__()
        self.name = name
        if name == "mse":
            self.base = nn.MSELoss()
            self.lpips = None
        elif name == "l1":
            self.base = nn.L1Loss()
            self.lpips = None
        elif name == "l1_lpips":
            _ensure_lpips_path()
            import lpips  # type: ignore

            self.base = nn.L1Loss()
            self.lpips = lpips.LPIPS(net=lpips_net, verbose=False).to(device)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            raise ValueError(name)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(pred, target)
        if self.lpips is None:
            return loss
        left, right = pred, target
        if left.shape[1] == 1:
            left = left.repeat(1, 3, 1, 1)
            right = right.repeat(1, 3, 1, 1)
        return loss + self.lpips(left, right).mean()
