from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .cases import CaseSpec
from .images import spec_acc, unit01_to_glow
from .util import ART_ROOT

if str(ART_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ART_ROOT / "scripts"))
import compute_victim_acc as acc  # noqa: E402


def glow_cli_args() -> argparse.Namespace:
    return argparse.Namespace(
        dither_mode=acc.DITHER_MODE,
        dither_thresh=acc.DITHER_THRESH,
        dither_eps_min=acc.DITHER_EPS_MIN,
        dither_eps_max=acc.DITHER_EPS_MAX,
        dither_seed=acc.DITHER_SEED,
        relu_patch_bits=acc.RELU_PATCH_BITS,
        relu_patch_fixed_bits=acc.RELU_PATCH_FIXED_BITS,
        relu_patch_fixed_value=acc.RELU_PATCH_FIXED_VALUE,
        relu_patch_inc=acc.RELU_PATCH_INC,
        relu_patch_positive=acc.RELU_PATCH_POSITIVE,
    )


def open_glow(art: Path, spec: CaseSpec, mode: str, work: Path) -> acc.StreamSession:
    bundle = art / "victims" / "glow" / "libs" / spec.case / mode
    runner = bundle / f"{spec.case}_runner"
    weights = bundle / f"{spec.case}.weights.bin"
    if not runner.is_file():
        raise FileNotFoundError(f"missing Glow runner {runner}")
    if not weights.is_file():
        raise FileNotFoundError(f"missing Glow weights {weights}")
    work.mkdir(parents=True, exist_ok=True)
    acc_spec = spec_acc(spec)
    return acc.StreamSession(
        [str(runner), str(weights), "--stream"],
        acc.glow_env(mode, glow_cli_args()),
        acc_spec.n_out,
        work / "stream.stderr",
    )


def infer_unit01(session: acc.StreamSession, image01: np.ndarray, spec: CaseSpec) -> np.ndarray:
    return session.infer(unit01_to_glow(image01, spec))


def pred_from_logits(logits: np.ndarray, spec: CaseSpec) -> int | np.ndarray:
    if spec.task == "multi_label":
        return acc.sigmoid(logits.astype(np.float32, copy=False))
    return int(np.argmax(logits))


class GlowBatch:
    """Holds an open stream so compute/train can query many images."""

    def __init__(self, art: Path, spec: CaseSpec, mode: str, work: Path):
        self.spec = spec
        self._session_cm = open_glow(art, spec, mode, work)
        self.session = None

    def __enter__(self) -> GlowBatch:
        self.session = self._session_cm.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        return self._session_cm.__exit__(*exc)

    def logits(self, image01: np.ndarray) -> np.ndarray:
        assert self.session is not None
        return infer_unit01(self.session, image01, self.spec)

    def logits_batch(self, images01: np.ndarray) -> np.ndarray:
        rows = [self.logits(images01[i]) for i in range(int(images01.shape[0]))]
        return np.stack(rows, axis=0)
