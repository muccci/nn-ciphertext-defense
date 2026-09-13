from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


ART_ROOT = Path(__file__).resolve().parents[2]
ATTACK_DIR = ART_ROOT / "attack"
COMMON_DIR = Path(__file__).resolve().parent
PRIORS_DIR = ATTACK_DIR / "priors"


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cipher_model():
    import importlib.util
    import sys

    path = COMMON_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("attack_cipher_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["attack_cipher_model"] = module
    spec.loader.exec_module(module)
    return module
