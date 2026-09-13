from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

from .cases import CaseSpec


def read_bits(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if raw and set(raw[: min(64, len(raw))]) <= {ord("0"), ord("1"), ord("\n"), ord("\r")}:
        text = raw.decode("ascii")
        values = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        values = values[(values == ord("0")) | (values == ord("1"))]
        return (values == ord("1")).astype(np.float32, copy=False)
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    return unpacked.astype(np.float32, copy=False)


def bits_to_trace(values: np.ndarray, spec: CaseSpec) -> torch.Tensor:
    take = values[: spec.bits]
    if int(take.shape[0]) < spec.bits:
        raise ValueError(f"{spec.case} bits got {take.shape[0]} want {spec.bits}")
    stride = max(1, int(spec.trace_stride))
    if stride > 1:
        take = take[::stride]
    if spec.fold is None:
        return torch.from_numpy(np.ascontiguousarray(take))
    channels, height, width = spec.fold
    need = channels * height * width
    pad = 1.0 if spec.pad_bit == "1" else 0.0
    flat = np.full((need,), pad, dtype=np.float32)
    used = min(int(take.shape[0]), need)
    flat[:used] = take[:used]
    return torch.from_numpy(np.ascontiguousarray(flat.reshape(spec.fold)))
