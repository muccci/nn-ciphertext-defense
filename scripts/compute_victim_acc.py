#!/usr/bin/env python3
"""Compute task utility for artifact victims (accuracy / top-1 / macro-AUROC).

Eval protocol:
  mnist / cifar        official test split (10000)
  imagenet50_96        2500 class-balanced, seed 1234
  celea / chest        5000 random without replacement, seed 1234
  preprocess           image_to_nchw (size / mean / std / gray→RGB)
  metrics              accuracy | top1_accuracy | macro_auroc (Mann–Whitney ranks)

Defense knobs are the same for MC / IC / TV / TA:
  ReLU patch 30 / 7 / 117 / inc 3 / positive 0
  input dither random, NCHW, seed 1234, thresh 1, eps 1e-5..2e-5
off = protect + dither disabled. on = those knobs + on products.
Requires --backend.

Usage:
  python3 scripts/compute_victim_acc.py --backend mc
  python3 scripts/compute_victim_acc.py --backend ic --case IC01
  python3 scripts/compute_victim_acc.py --backend tv --modes off,on
  python3 scripts/compute_victim_acc.py --backend ta --case TA01

Host extras: numpy, Pillow, torchvision (MNIST / CIFAR adapters).
Datasets: --dataset-root or ACC_TEST_DATASET_ROOT.
ImageNet50_96: --imagenet50-96-root or ACC_TEST_IMAGENET50_96_ROOT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

BACKEND_PREFIX = {"mc": "MC", "ic": "IC", "tv": "TV", "ta": "TA"}
FORCE_SINGLE_SAMPLE_FILE = frozenset({"squeezenet", "densenet"})

# Unified Def+ — same keys as victims/{glow,tvm} compile/collect env.
RELU_PATCH_BITS = 30
RELU_PATCH_FIXED_BITS = 7
RELU_PATCH_FIXED_VALUE = 117  # 0x75
RELU_PATCH_INC = 3
RELU_PATCH_POSITIVE = 0
DITHER_MODE = "random"
DITHER_LAYOUT = "NCHW"
DITHER_THRESH = 1.0
DITHER_EPS_MIN = 1e-5
DITHER_EPS_MAX = 2e-5
DITHER_SEED = 1234
SUBSET_SEED = 1234
IMAGENET50_COUNT = 2500
RANDOM_SUBSET_COUNT = 5000

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
HALF_RGB_MEAN = (0.5, 0.5, 0.5)
HALF_RGB_STD = (0.5, 0.5, 0.5)
HALF_GRAY_MEAN = (0.5,)
HALF_GRAY_STD = (0.5,)
CHEST_1CH_MEAN = (0.449,)
CHEST_1CH_STD = (0.226,)
CHEST_LABELS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
)
IC_LD_LIBRARY_PATH = os.environ.get(
    "ACC_TEST_RUNTIME_LD_LIBRARY_PATH",
    "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
)


@dataclass(frozen=True)
class DataSpec:
    family: str
    dataset: str
    source: str
    task: str
    metric: str
    height: int
    width: int
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    n_out: int
    gray_to_rgb: bool = False
    subset: str = "full"  # full | balanced2500 | random5000


SPECS: dict[tuple[str, str], DataSpec] = {}


def _spec(spec: DataSpec) -> None:
    SPECS[(spec.family, spec.dataset)] = spec


_spec(DataSpec("lenet", "mnist", "mnist", "single_label", "accuracy", 28, 28, 1, MNIST_MEAN, MNIST_STD, 10))
_spec(DataSpec("lenet", "mnist64", "mnist", "single_label", "accuracy", 64, 64, 1, MNIST_MEAN, MNIST_STD, 10))
_spec(DataSpec("vggnet", "mnist", "mnist", "single_label", "accuracy", 28, 28, 1, MNIST_MEAN, MNIST_STD, 10))
_spec(DataSpec("vggnet", "cifar", "cifar10", "single_label", "accuracy", 32, 32, 3, CIFAR10_MEAN, CIFAR10_STD, 10))
_spec(DataSpec("squeezenet", "cifar", "cifar10", "single_label", "accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 10))
_spec(DataSpec("squeezenet", "cifar32", "cifar10", "single_label", "accuracy", 32, 32, 3, IMAGENET_MEAN, IMAGENET_STD, 10))
_spec(DataSpec("squeezenet", "imagenet50_96", "imagenet50_96", "single_label", "top1_accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 50, subset="balanced2500"))
_spec(DataSpec("resnet", "mnist", "mnist", "single_label", "accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 10, gray_to_rgb=True))
_spec(DataSpec("resnet", "cifar", "cifar10", "single_label", "accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 10))
_spec(DataSpec("resnet", "imagenet50_96", "imagenet50_96", "single_label", "top1_accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 50, subset="balanced2500"))
_spec(DataSpec("resnet", "celea", "celeba", "single_label", "accuracy", 224, 224, 3, IMAGENET_MEAN, IMAGENET_STD, 10177, subset="random5000"))
_spec(DataSpec("resnet", "chest", "chestxray14", "multi_label", "macro_auroc", 224, 224, 3, IMAGENET_MEAN, IMAGENET_STD, 14, gray_to_rgb=True, subset="random5000"))
_spec(DataSpec("resnet", "chest_single_channel", "chestxray14", "multi_label", "macro_auroc", 224, 224, 1, CHEST_1CH_MEAN, CHEST_1CH_STD, 14, subset="random5000"))
_spec(DataSpec("mobilenet", "mnist", "mnist", "single_label", "accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 10, gray_to_rgb=True))
_spec(DataSpec("mobilenet", "cifar", "cifar10", "single_label", "accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 10))
_spec(DataSpec("mobilenet", "celea", "celeba", "single_label", "accuracy", 224, 224, 3, IMAGENET_MEAN, IMAGENET_STD, 10177, subset="random5000"))
_spec(DataSpec("mobilenet", "chest", "chestxray14", "multi_label", "macro_auroc", 224, 224, 3, IMAGENET_MEAN, IMAGENET_STD, 14, gray_to_rgb=True, subset="random5000"))
_spec(DataSpec("densenet", "imagenet50_96", "imagenet50_96", "single_label", "top1_accuracy", 96, 96, 3, IMAGENET_MEAN, IMAGENET_STD, 50, subset="balanced2500"))
_spec(DataSpec("densenet", "celea", "celeba", "single_label", "accuracy", 218, 178, 3, HALF_RGB_MEAN, HALF_RGB_STD, 10177, subset="random5000"))
_spec(DataSpec("densenet", "chest", "chestxray14", "multi_label", "macro_auroc", 224, 224, 1, HALF_GRAY_MEAN, HALF_GRAY_STD, 14, subset="random5000"))


def parse_int_auto(value: str) -> int:
    return int(value, 0)


def derive_subset_seed(base_seed: int, dataset_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{dataset_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def resolve_dataset_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli
    env = os.environ.get("ACC_TEST_DATASET_ROOT")
    if env:
        return Path(env)
    art_data = Path(__file__).resolve().parent.parent / "data"
    if (art_data / "mnist").is_dir():
        return art_data
    raise SystemExit("set --dataset-root or ACC_TEST_DATASET_ROOT")


def resolve_imagenet50_root(cli: Path | None, data_root: Path) -> Path:
    if cli is not None:
        return cli
    env = os.environ.get("ACC_TEST_IMAGENET50_96_ROOT")
    if env:
        return Path(env)
    return data_root / "imagenet50_96"


def image_to_nchw(image: Image.Image, spec: DataSpec) -> np.ndarray:
    if spec.channels == 1:
        image = image.convert("L")
    elif spec.gray_to_rgb:
        image = image.convert("L").convert("RGB")
    else:
        image = image.convert("RGB")
    if image.size != (spec.width, spec.height):
        image = image.resize((spec.width, spec.height), RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if spec.channels == 1:
        array = array[np.newaxis, :, :]
    else:
        array = np.transpose(array, (2, 0, 1))
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(-1, 1, 1)
    array = (array - mean) / std
    return array[np.newaxis, ...].astype(np.float32, copy=False)


class Adapter:
    def __len__(self) -> int:
        raise NotImplementedError

    def get(self, index: int) -> tuple[Any, Any]:
        raise NotImplementedError

    def labels(self) -> list[int] | None:
        return None


class MNISTAdapter(Adapter):
    def __init__(self, root: Path):
        from torchvision import datasets
        self.ds = datasets.MNIST(root=str(root), train=False, download=False)

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, index: int) -> tuple[Image.Image, int]:
        image, label = self.ds[index]
        return image, int(label)

    def labels(self) -> list[int]:
        return [int(x) for x in self.ds.targets]


class CIFAR10Adapter(Adapter):
    def __init__(self, root: Path):
        from torchvision import datasets
        self.ds = datasets.CIFAR10(root=str(root), train=False, download=False)

    def __len__(self) -> int:
        return len(self.ds)

    def get(self, index: int) -> tuple[Image.Image, int]:
        image, label = self.ds[index]
        return image, int(label)

    def labels(self) -> list[int]:
        return [int(x) for x in self.ds.targets]


def compute_split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 1:
        return n, 0, 0
    if n == 2:
        return 1, 1, 0
    val_count = max(1, int(round(n * val_ratio)))
    test_count = max(1, int(round(n * test_ratio)))
    while n - val_count - test_count < 1:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    train_count = n - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    return train_count, val_count, test_count


class CelebAAdapter(Adapter):
    def __init__(self, root: Path):
        image_root = root / "celeba" / "Dataset" / "CelebA_train" / "img_align_celeba"
        identity_file = root / "celeba" / "annotations" / "identity_CelebA.txt"
        if not identity_file.is_file():
            raise FileNotFoundError(f"missing CelebA identity file: {identity_file}")
        by_id: dict[int, list[str]] = {}
        for line in identity_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            filename, identity = line.split()
            path = image_root / filename
            if path.is_file():
                by_id.setdefault(int(identity) - 1, []).append(str(path))
        if not by_id:
            raise FileNotFoundError(f"no CelebA images under {image_root}")
        samples: list[tuple[str, int]] = []
        rng = random.Random(1234)
        for label in sorted(by_id):
            paths = list(by_id[label])
            rng.shuffle(paths)
            train_n, val_n, test_n = compute_split_counts(len(paths), 0.1, 0.1)
            start = train_n + val_n
            samples.extend((p, label) for p in paths[start:start + test_n])
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), int(label)

    def labels(self) -> list[int]:
        return [int(label) for _, label in self.samples]


class ChestAdapter(Adapter):
    def __init__(self, root: Path):
        data_dir = root / "chestxray14"
        self.image_dir = data_dir / "images"
        list_path = data_dir / "test_list.txt"
        csv_path = data_dir / "Data_Entry_2017_v2020.csv"
        if not list_path.is_file():
            raise FileNotFoundError(f"missing ChestX-ray14 list: {list_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing ChestX-ray14 csv: {csv_path}")
        self.names = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ann: dict[str, np.ndarray] = {}
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target = np.zeros(len(CHEST_LABELS), dtype=np.float32)
                labels = row["Finding Labels"]
                if labels != "No Finding":
                    for label in labels.split("|"):
                        if label in CHEST_LABELS:
                            target[CHEST_LABELS.index(label)] = 1.0
                ann[row["Image Index"]] = target
        self.ann = {name: ann[name] for name in self.names}

    def __len__(self) -> int:
        return len(self.names)

    def get(self, index: int) -> tuple[Image.Image, np.ndarray]:
        name = self.names[index]
        with Image.open(self.image_dir / name) as image:
            return image.convert("L"), self.ann[name].copy()


class ImageFolderAdapter(Adapter):
    def __init__(self, root: Path):
        if not root.is_dir():
            raise FileNotFoundError(f"missing ImageNet50_96 root: {root}")
        label_map = json.loads((root / "label_map.json").read_text(encoding="utf-8"))
        class_dirs = [str(x) for x in label_map["class_dirs"]]
        split_root = root / "validation"
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing ImageNet50_96 split: {split_root}")
        samples: list[tuple[Path, int]] = []
        for label, class_dir in enumerate(class_dirs):
            class_root = split_root / class_dir
            if not class_root.is_dir():
                raise FileNotFoundError(f"missing ImageNet50_96 class: {class_root}")
            for path in sorted(p for p in class_root.iterdir() if p.is_file()):
                samples.append((path, label))
        if not samples:
            raise RuntimeError(f"no ImageNet50_96 files under {split_root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), int(label)

    def labels(self) -> list[int]:
        return [int(label) for _, label in self.samples]


def build_adapter(spec: DataSpec, data_root: Path, imagenet_root: Path) -> Adapter:
    if spec.source == "mnist":
        return MNISTAdapter(data_root / "mnist")
    if spec.source == "cifar10":
        return CIFAR10Adapter(data_root / "cifar10")
    if spec.source == "celeba":
        return CelebAAdapter(data_root)
    if spec.source == "chestxray14":
        return ChestAdapter(data_root)
    if spec.source == "imagenet50_96":
        return ImageFolderAdapter(imagenet_root)
    raise RuntimeError(f"unsupported dataset source: {spec.source}")


def load_subset_manifest(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload)
    if not isinstance(datasets, dict):
        raise RuntimeError(f"subset manifest datasets must be an object: {path}")
    out: dict[str, list[int]] = {}
    for raw_name, entry in datasets.items():
        indices = entry["indices"] if isinstance(entry, dict) else entry
        if not isinstance(indices, list):
            raise RuntimeError(f"subset indices for {raw_name} must be a list")
        out[str(raw_name).lower()] = [int(x) for x in indices]
    return out


def balanced_subset(labels: list[int], count: int, seed: int, dataset_name: str) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(i)
    class_ids = sorted(by_class)
    if not class_ids:
        raise RuntimeError(f"no classes for balanced subset of {dataset_name}")
    base = count // len(class_ids)
    extra = count % len(class_ids)
    if base == 0:
        raise RuntimeError(f"balanced count {count} < n_classes {len(class_ids)}")
    rng = random.Random(derive_subset_seed(seed, dataset_name))
    extra_classes = set(rng.sample(class_ids, extra)) if extra else set()
    picked: list[int] = []
    for class_id in class_ids:
        want = base + (1 if class_id in extra_classes else 0)
        pool = list(by_class[class_id])
        if want > len(pool):
            raise RuntimeError(f"class {class_id} has {len(pool)} < {want}")
        rng.shuffle(pool)
        picked.extend(pool[:want])
    picked.sort()
    if len(picked) != count:
        raise RuntimeError(f"balanced subset size {len(picked)} != {count}")
    return picked


def select_indices(
    spec: DataSpec,
    adapter: Adapter,
    limit: int | None,
    manifest_indices: dict[str, list[int]],
) -> list[int]:
    n = len(adapter)
    if spec.dataset in manifest_indices:
        indices = list(manifest_indices[spec.dataset])
        for idx in indices:
            if idx < 0 or idx >= n:
                raise RuntimeError(f"{spec.dataset} subset index {idx} not in [0, {n})")
    elif spec.subset == "full":
        indices = list(range(n))
    elif spec.subset == "balanced2500":
        labels = adapter.labels()
        if labels is None:
            raise RuntimeError(f"{spec.dataset} has no class labels for balanced subset")
        indices = balanced_subset(labels, IMAGENET50_COUNT, SUBSET_SEED, spec.dataset)
    elif spec.subset == "random5000":
        k = min(RANDOM_SUBSET_COUNT, n)
        rng = random.Random(derive_subset_seed(SUBSET_SEED, spec.dataset))
        indices = sorted(rng.sample(range(n), k))
    else:
        raise RuntimeError(spec.subset)
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        raise RuntimeError(f"no samples selected for {spec.dataset}")
    return indices


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    ranks = np.arange(1, len(sorted_scores) + 1, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    sum_pos = float(ranks[sorted_true.astype(bool)].sum())
    return (sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        auc = binary_auc(y_true[:, i], y_prob[:, i])
        if not math.isnan(auc):
            scores.append(auc)
    return float(np.mean(scores)) if scores else float("nan")


class StreamSession:
    def __init__(self, cmd: list[str], env: dict[str, str], n_out: int, stderr_path: Path):
        self.cmd = cmd
        self.env = env
        self.out_bytes = n_out * 4
        self.stderr_path = stderr_path
        self.proc: subprocess.Popen[bytes] | None = None
        self.err: Any = None

    def __enter__(self) -> StreamSession:
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.err = self.stderr_path.open("wb")
        env = os.environ.copy()
        env.update(self.env)
        env.setdefault("GLOW_INPUT_ZERO_DITHER_SILENT", "1")
        env.setdefault("TVM_INPUT_ZERO_DITHER_SILENT", "1")
        self.proc = subprocess.Popen(
            self.cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.err, bufsize=0,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"failed to open pipes: {self.cmd}")
        return self

    def __exit__(self, exc_type: Any, *_exc: Any) -> bool:
        if self.proc is not None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except BrokenPipeError:
                pass
            rc = self.proc.wait()
            if rc != 0 and exc_type is None:
                raise RuntimeError(f"runner exit {rc}: {self.cmd}\n{self._stderr()}")
        if self.err is not None:
            self.err.close()
        return False

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        assert self.proc and self.proc.stdin and self.proc.stdout
        payload = np.ascontiguousarray(tensor, dtype=np.float32)
        try:
            self.proc.stdin.write(payload.tobytes(order="C"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(f"stdin closed: {self.cmd}\n{self._stderr()}") from exc
        chunks: list[bytes] = []
        left = self.out_bytes
        while left > 0:
            chunk = self.proc.stdout.read(left)
            if not chunk:
                raise RuntimeError(
                    f"short read {self.out_bytes - left}/{self.out_bytes}: {self.cmd}\n{self._stderr()}"
                )
            chunks.append(chunk)
            left -= len(chunk)
        return np.frombuffer(b"".join(chunks), dtype=np.float32).copy()

    def _stderr(self) -> str:
        if not self.stderr_path.exists():
            return ""
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")


def glow_env(mode: str, args: argparse.Namespace) -> dict[str, str]:
    if mode == "off":
        return {
            "GLOW_WRITEBACK_PROTECT": "0",
            "GLOW_RELU_LOW12_PATCH": "0",
            "GLOW_MAXPOOL_LOWBIT_PATCH": "0",
            "GLOW_INPUT_ZERO_DITHER": "0",
        }
    return {
        "GLOW_WRITEBACK_PROTECT": "1",
        "GLOW_RELU_LOW12_PATCH": "1",
        "GLOW_MAXPOOL_LOWBIT_PATCH": "1",
        "GLOW_INPUT_ZERO_DITHER": args.dither_mode,
        "GLOW_INPUT_ZERO_DITHER_LAYOUT": DITHER_LAYOUT,
        "GLOW_INPUT_ZERO_DITHER_THRESH": str(args.dither_thresh),
        "GLOW_INPUT_ZERO_DITHER_EPS_MIN": str(args.dither_eps_min),
        "GLOW_INPUT_ZERO_DITHER_EPS_MAX": str(args.dither_eps_max),
        "GLOW_INPUT_ZERO_DITHER_SEED": str(args.dither_seed),
        "GLOW_RELU_PATCH_BITS": str(args.relu_patch_bits),
        "GLOW_RELU_PATCH_FIXED_BITS": str(args.relu_patch_fixed_bits),
        "GLOW_RELU_PATCH_FIXED_VALUE": str(args.relu_patch_fixed_value),
        "GLOW_RELU_PATCH_INC": str(args.relu_patch_inc),
        "GLOW_RELU_PATCH_POSITIVE": str(args.relu_patch_positive),
    }


def tvm_env(mode: str, args: argparse.Namespace) -> dict[str, str]:
    if mode == "off":
        return {
            "TVM_WRITEBACK_PROTECT": "0",
            "GLOW_WRITEBACK_PROTECT": "0",
            "TVM_RELU_LOW12_PATCH": "0",
            "TVM_INPUT_ZERO_DITHER": "0",
            "GLOW_INPUT_ZERO_DITHER": "0",
        }
    return {
        **glow_env("on", args),
        "TVM_WRITEBACK_PROTECT": "1",
        "TVM_RELU_LOW12_PATCH": "1",
        "TVM_RELU_PATCH_BITS": str(args.relu_patch_bits),
        "TVM_RELU_PATCH_FIXED_BITS": str(args.relu_patch_fixed_bits),
        "TVM_RELU_PATCH_FIXED_VALUE": hex(args.relu_patch_fixed_value),
        "TVM_RELU_PATCH_INC": str(args.relu_patch_inc),
        "TVM_RELU_PATCH_POSITIVE": str(args.relu_patch_positive),
        "TVM_INPUT_ZERO_DITHER": args.dither_mode,
        "TVM_INPUT_ZERO_DITHER_LAYOUT": DITHER_LAYOUT,
        "TVM_INPUT_ZERO_DITHER_THRESH": str(args.dither_thresh),
        "TVM_INPUT_ZERO_DITHER_EPS_MIN": str(args.dither_eps_min),
        "TVM_INPUT_ZERO_DITHER_EPS_MAX": str(args.dither_eps_max),
        "TVM_INPUT_ZERO_DITHER_SEED": str(args.dither_seed),
    }


def load_manifest(manifest: Path, backend: str) -> dict[str, dict[str, str]]:
    prefix = BACKEND_PREFIX[backend]
    rows: dict[str, dict[str, str]] = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].startswith("MC"):
            continue
        mc, task, onnx_rel = parts[0], parts[1], parts[3]
        case_id = f"{prefix}{mc[2:]}"
        rows[case_id] = {
            "task": task,
            "model": onnx_rel.split("/")[0],
            "family": onnx_rel.split("/")[0].lower(),
            "dataset": onnx_rel.split("/")[1],
            "onnx_rel": onnx_rel,
        }
    return rows


def spec_for(row: dict[str, str]) -> DataSpec:
    key = (row["family"], row["dataset"])
    if key not in SPECS:
        raise SystemExit(f"no acc spec for {row['family']}/{row['dataset']}")
    return SPECS[key]


def eval_stream(
    adapter: Adapter,
    spec: DataSpec,
    indices: list[int],
    session: StreamSession,
) -> dict[str, Any]:
    n = len(indices)
    if spec.task == "single_label":
        correct = 0
        for i, idx in enumerate(indices, 1):
            image, label = adapter.get(idx)
            pred = int(np.argmax(session.infer(image_to_nchw(image, spec))))
            correct += int(pred == int(label))
            if i % 100 == 0 or i == n:
                print(f"  processed {i}/{n}", flush=True)
        return {
            "metric": spec.metric,
            "value": correct / max(n, 1),
            "correct": correct,
            "n": n,
        }
    y_true = np.zeros((n, spec.n_out), dtype=np.float32)
    y_prob = np.zeros((n, spec.n_out), dtype=np.float32)
    for i, idx in enumerate(indices):
        image, target = adapter.get(idx)
        logits = session.infer(image_to_nchw(image, spec))
        y_true[i] = np.asarray(target, dtype=np.float32)
        y_prob[i] = sigmoid(logits.astype(np.float32, copy=False))
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}", flush=True)
    return {"metric": spec.metric, "value": macro_auroc(y_true, y_prob), "n": n}


def eval_logits(spec: DataSpec, logits: np.ndarray, targets: Any) -> dict[str, Any]:
    if spec.task == "single_label":
        pred = logits.argmax(axis=1)
        tgt = np.asarray(targets, dtype=np.int64)
        correct = int((pred == tgt).sum())
        return {
            "metric": spec.metric,
            "value": correct / max(len(tgt), 1),
            "correct": correct,
            "n": int(len(tgt)),
        }
    y_true = np.asarray(targets, dtype=np.float32)
    y_prob = sigmoid(logits.astype(np.float32, copy=False))
    return {"metric": spec.metric, "value": macro_auroc(y_true, y_prob), "n": int(y_true.shape[0])}


def run_mc(
    art: Path,
    case_id: str,
    spec: DataSpec,
    adapter: Adapter,
    indices: list[int],
    mode: str,
    args: argparse.Namespace,
    work: Path,
) -> dict[str, Any]:
    bundle = art / "victims" / "glow" / "libs" / case_id / mode
    runner = bundle / f"{case_id}_runner"
    weights = bundle / f"{case_id}.weights.bin"
    if not runner.is_file():
        raise FileNotFoundError(f"missing MC runner: {runner}")
    if not weights.is_file():
        raise FileNotFoundError(f"missing MC weights: {weights}")
    work.mkdir(parents=True, exist_ok=True)
    cmd = [str(runner), str(weights), "--stream"]
    with StreamSession(cmd, glow_env(mode, args), spec.n_out, work / "stream.stderr") as session:
        return eval_stream(adapter, spec, indices, session)


def run_ic(
    art: Path,
    row: dict[str, str],
    spec: DataSpec,
    adapter: Adapter,
    indices: list[int],
    mode: str,
    args: argparse.Namespace,
    work: Path,
) -> dict[str, Any]:
    ic_bin = args.ic_bin or Path(
        os.environ.get("IC_BIN", art / "writeback-protect/glow/build_ic/bin/image-classifier")
    )
    onnx = art / "victims" / "models" / row["onnx_rel"]
    if not ic_bin.is_file():
        raise FileNotFoundError(f"missing image-classifier: {ic_bin}")
    if not onnx.is_file():
        raise FileNotFoundError(f"missing ONNX: {onnx}")
    work.mkdir(parents=True, exist_ok=True)
    per_file = 1 if row["family"] in FORCE_SINGLE_SAMPLE_FILE else 100
    batch_dir = work / "inputs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    targets: list[Any] = []
    for start in range(0, len(indices), per_file):
        chunk = indices[start:start + per_file]
        tensors = []
        for idx in chunk:
            image, target = adapter.get(idx)
            tensors.append(image_to_nchw(image, spec))
            targets.append(target)
        batch = np.concatenate(tensors, axis=0).astype(np.float32, copy=False)
        path = batch_dir / f"batch_{start // per_file:04d}.npy"
        np.save(path, batch)
        paths.append(str(path))
        print(f"  prepared {min(start + per_file, len(indices))}/{len(indices)}", flush=True)
    list_path = batch_dir / "input_batches.list.txt"
    list_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
    raw_out = work / "raw_outputs.bin"
    if raw_out.exists():
        raw_out.unlink()
    cmd = [
        str(ic_bin),
        f"-model={onnx}",
        "-backend=Interpreter",
        "-model-input=data",
        f"-input-image-list-file={list_path}",
        "-output-name=output",
        "-image-layout=NCHW",
        "-input-layout=NCHW",
        "-minibatch=1",
        "-minibatch-threads=1",
        "-topk=0",
        f"-dump-output-binary-file={raw_out}",
    ]
    env = os.environ.copy()
    env.update(glow_env(mode, args))
    env["LD_LIBRARY_PATH"] = IC_LD_LIBRARY_PATH
    env.setdefault("GLOW_INPUT_ZERO_DITHER_SILENT", "1")
    completed = subprocess.run(cmd, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (work / "stdout.txt").write_bytes(completed.stdout)
    (work / "stderr.txt").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"image-classifier failed ({completed.returncode})\n"
            f"{completed.stderr.decode('utf-8', 'replace')}"
        )
    flat = np.fromfile(raw_out, dtype=np.float32)
    expected = len(indices) * spec.n_out
    if flat.size != expected:
        raise RuntimeError(f"IC output size {flat.size} != {expected}")
    return eval_logits(spec, flat.reshape(len(indices), spec.n_out), targets)


def run_tvm(
    art: Path,
    case_id: str,
    spec: DataSpec,
    adapter: Adapter,
    indices: list[int],
    mode: str,
    backend: str,
    args: argparse.Namespace,
    work: Path,
) -> dict[str, Any]:
    tvm_dir = art / "victims" / "tvm"
    tvm_build = args.tvm_build or (art / "writeback-protect" / "tvm" / "build-llvm14-glow")
    libs = tvm_dir / "libs" / case_id
    if backend == "tv":
        runner = tvm_dir / "bin" / "generic_tvm_native_vm_runner"
        library = libs / mode / f"{case_id}_tvm.so"
        shape = (libs / mode / "input_shape.txt").read_text().strip()
        cmd = [str(runner), "--library", str(library), "--input-shape", shape, "--stream"]
        extras = [runner, library, libs / mode / "input_shape.txt"]
    else:
        runner = libs / f"{case_id}_tvm_aot_runner"
        library = libs / mode / f"{case_id}_tvm_aot_kernels.so"
        constants = libs / mode / "constants"
        cmd = [
            str(runner), "--library", str(library),
            "--constants-dir", str(constants), "--stream",
        ]
        extras = [runner, library, constants]
    for path in extras:
        if not path.exists():
            raise FileNotFoundError(path)
    work.mkdir(parents=True, exist_ok=True)
    env = tvm_env(mode, args)
    extra_ld = f"{tvm_build}:{tvm_build / 'lib'}"
    env["LD_LIBRARY_PATH"] = extra_ld + (
        (":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else ""
    )
    with StreamSession(cmd, env, spec.n_out, work / "stream.stderr") as session:
        return eval_stream(adapter, spec, indices, session)


def normalize_cases(raw: list[str], backend: str, manifest_ids: list[str]) -> list[str]:
    prefix = BACKEND_PREFIX[backend]
    if not raw:
        return manifest_ids
    out: list[str] = []
    for item in raw:
        if item.startswith(("MC", "IC", "TV", "TA")) and len(item) >= 4:
            out.append(f"{prefix}{item[2:]}")
        else:
            out.append(f"{prefix}{item.zfill(2)}")
    return out


def fmt_metric(value: float) -> str:
    return "nan" if value != value else f"{value:.6f}"


def main() -> int:
    art = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=("mc", "ic", "tv", "ta"), required=True)
    parser.add_argument(
        "--case", action="append", default=[], dest="cases",
        help="MC01 / IC01 / TV01 / TA01 (repeatable). Default: all from manifest.",
    )
    parser.add_argument("--modes", default="off,on", help="comma list: off,on")
    parser.add_argument("--manifest", type=Path, default=art / "victims/models/MANIFEST.txt")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--imagenet50-96-root", type=Path, default=None)
    parser.add_argument(
        "--subset-manifest", type=Path, default=None,
        help="optional acc_test subset JSON (overrides built-in split for listed datasets)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap samples (debug). Default: full eval split.")
    parser.add_argument("--out-tsv", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--ic-bin", type=Path, default=None)
    parser.add_argument("--tvm-build", type=Path, default=None)
    parser.add_argument("--relu-patch-bits", type=parse_int_auto, default=RELU_PATCH_BITS)
    parser.add_argument("--relu-patch-fixed-bits", type=parse_int_auto, default=RELU_PATCH_FIXED_BITS)
    parser.add_argument("--relu-patch-fixed-value", type=parse_int_auto, default=RELU_PATCH_FIXED_VALUE)
    parser.add_argument("--relu-patch-inc", type=parse_int_auto, default=RELU_PATCH_INC)
    parser.add_argument("--relu-patch-positive", type=parse_int_auto, default=RELU_PATCH_POSITIVE)
    parser.add_argument("--dither-mode", default=DITHER_MODE)
    parser.add_argument("--dither-thresh", type=float, default=DITHER_THRESH)
    parser.add_argument("--dither-eps-min", type=float, default=DITHER_EPS_MIN)
    parser.add_argument("--dither-eps-max", type=float, default=DITHER_EPS_MAX)
    parser.add_argument("--dither-seed", type=parse_int_auto, default=DITHER_SEED)
    args = parser.parse_args()

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not modes or any(mode not in {"off", "on"} for mode in modes):
        raise SystemExit("mode must be off|on")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be > 0")

    data_root = resolve_dataset_root(args.dataset_root)
    imagenet_root = resolve_imagenet50_root(args.imagenet50_96_root, data_root)
    work_root = args.work_root or (art / "acc_out" / args.backend)
    meta = load_manifest(args.manifest, args.backend)
    case_ids = normalize_cases(args.cases, args.backend, sorted(meta))
    unknown = [case_id for case_id in case_ids if case_id not in meta]
    if unknown:
        raise SystemExit(f"unknown case(s) for --backend {args.backend}: {unknown}")
    subset_indices = load_subset_manifest(args.subset_manifest) if args.subset_manifest else {}

    print(f"backend={args.backend}")
    print(f"dataset_root={data_root}")
    print(f"{args.backend}\ttask\tdataset\tmetric\tn\t" + "\t".join(modes))

    rows: list[dict[str, Any]] = []
    missing = 0
    for case_id in case_ids:
        info = meta[case_id]
        spec = spec_for(info)
        print(f"=== {case_id} {info['family']}/{info['dataset']} ===", flush=True)
        try:
            adapter = build_adapter(spec, data_root, imagenet_root)
            indices = select_indices(spec, adapter, args.limit, subset_indices)
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            print(f"{case_id}\tmissing\t{exc}")
            rows.append({
                args.backend: case_id,
                "task": info["task"],
                "model": info["model"],
                "dataset": info["dataset"],
                "metric": spec.metric,
                "n": "",
                "status": "missing",
                "error": str(exc),
            })
            missing += 1
            continue

        values: dict[str, float] = {}
        n = len(indices)
        status = "ok"
        for mode in modes:
            work = work_root / case_id / mode
            try:
                if args.backend == "mc":
                    result = run_mc(art, case_id, spec, adapter, indices, mode, args, work)
                elif args.backend == "ic":
                    result = run_ic(art, info, spec, adapter, indices, mode, args, work)
                else:
                    result = run_tvm(
                        art, case_id, spec, adapter, indices, mode, args.backend, args, work,
                    )
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"{case_id}/{mode}\tmissing\t{exc}")
                values[mode] = float("nan")
                status = "partial"
                missing += 1
                continue
            values[mode] = float(result["value"])
            n = int(result["n"])

        line = [
            case_id, info["task"], f"{info['family']}/{info['dataset']}",
            spec.metric, str(n),
        ]
        line.extend(fmt_metric(values.get(mode, float("nan"))) for mode in modes)
        print("\t".join(line))
        row: dict[str, Any] = {
            args.backend: case_id,
            "task": info["task"],
            "model": info["model"],
            "dataset": info["dataset"],
            "metric": spec.metric,
            "n": n,
            "status": status,
        }
        for mode in modes:
            row[mode] = values.get(mode, float("nan"))
        if "off" in values and "on" in values and values["off"] == values["off"]:
            row["delta"] = values["on"] - values["off"]
        rows.append(row)

    out_tsv = args.out_tsv or (work_root / "acc_results.tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [args.backend, "task", "model", "dataset", "metric", "n", *modes, "delta", "status", "error"]
    with out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_tsv}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
