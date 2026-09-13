from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .cases import CaseSpec
from .util import ART_ROOT

if str(ART_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ART_ROOT / "scripts"))
import compute_victim_acc as acc  # noqa: E402


def spec_acc(case_spec: CaseSpec):
    return acc.SPECS[(case_spec.family, case_spec.dataset)]


def victim_to_model(image01: torch.Tensor) -> torch.Tensor:
    return image01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def model_to_victim(image_model: torch.Tensor) -> torch.Tensor:
    return image_model.add(1.0).mul(0.5).clamp(0.0, 1.0)


def load_nchw_f32(path: Path, spec: CaseSpec) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    want = spec.nc * spec.height * spec.width
    if raw.size != want:
        raise ValueError(f"{path} nchw size {raw.size} want {want}")
    return raw.reshape(spec.nc, spec.height, spec.width)


def nchw_to_unit01(nchw: np.ndarray, spec: CaseSpec) -> np.ndarray:
    """Glow bin may be unit [0,1] or already mean/std-normalized."""
    vmin = float(nchw.min())
    vmax = float(nchw.max())
    if vmin >= -0.02 and vmax <= 1.02:
        return np.clip(nchw, 0.0, 1.0).astype(np.float32, copy=False)
    acc_spec = spec_acc(spec)
    mean = np.asarray(acc_spec.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(acc_spec.std, dtype=np.float32).reshape(-1, 1, 1)
    return np.clip(nchw * std + mean, 0.0, 1.0).astype(np.float32, copy=False)


def unit01_to_glow(image01: np.ndarray, spec: CaseSpec) -> np.ndarray:
    acc_spec = spec_acc(spec)
    if spec.nc == 1:
        plane = np.clip(image01[0] * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(plane, mode="L")
    else:
        hw = np.clip(np.transpose(image01, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(hw, mode="RGB")
    return acc.image_to_nchw(pil, acc_spec)


def save_unit01_png(path: Path, image01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image01.shape[0] == 1:
        plane = np.clip(image01[0] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(plane, mode="L").save(path)
        return
    hw = np.clip(np.transpose(image01, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(hw, mode="RGB").save(path)
