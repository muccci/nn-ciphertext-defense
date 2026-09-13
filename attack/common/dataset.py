from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cases import CaseSpec
from .images import load_nchw_f32, nchw_to_unit01
from .traces import bits_to_trace, read_bits
from .util import ART_ROOT


@dataclass
class Sample:
    sample_id: str
    split: str
    bits_path: Path
    image_path: Path | None
    label: np.ndarray
    patient_id: int | None = None


def _read_meta(path: Path, spec: CaseSpec) -> np.ndarray | None:
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if spec.task == "multi_label":
        if "gt_positive_indices" in meta:
            out = np.zeros((spec.n_out,), dtype=np.float32)
            for idx in meta["gt_positive_indices"]:
                out[int(idx)] = 1.0
            return out
        if "label" in meta and isinstance(meta["label"], (list, tuple)):
            return np.asarray(meta["label"], dtype=np.float32)
        return None
    for key in ("label", "label_index", "gt_label"):
        if key in meta:
            return np.asarray(int(meta[key]), dtype=np.int64)
    return None


def _empty_label(spec: CaseSpec) -> np.ndarray:
    if spec.task == "multi_label":
        return np.zeros((spec.n_out,), dtype=np.float32)
    return np.asarray(-1, dtype=np.int64)


def _add_record(
    records: list[Sample],
    spec: CaseSpec,
    bits_path: Path,
    split: str,
    sample_id: str,
    default_image: Path | None,
) -> None:
    parent = bits_path.parent
    image = None
    for name in ("input_nchw_f32.bin", "image_nchw_f32.bin"):
        cand = parent / name
        if cand.is_file():
            image = cand
            break
    if image is None:
        image = default_image
    meta_path = parent / "meta.json"
    label = _read_meta(meta_path, spec)
    if label is None:
        label = _empty_label(spec)
    patient_id = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("patient_id", "patient", "pid"):
            if key in meta:
                patient_id = int(meta[key])
                break
    records.append(
        Sample(
            sample_id=sample_id,
            split=split,
            bits_path=bits_path,
            image_path=image,
            label=label,
            patient_id=patient_id,
        )
    )


def discover_samples(
    spec: CaseSpec,
    mode: str,
    traces: Path | None,
    *,
    input_bin: Path | None = None,
) -> list[Sample]:
    root = traces
    if root is None:
        root = ART_ROOT / "trace_out" / "attack" / spec.case / mode / spec.site
    root = Path(root)
    records: list[Sample] = []
    default_image = input_bin
    if default_image is None:
        bundled = (
            ART_ROOT
            / "victims"
            / "models"
            / spec.family
            / spec.dataset
            / "input_nchw_f32.bin"
        )
        if bundled.is_file():
            default_image = bundled

    bit_names = ("bits.txt", "bits.bin.gz", "bits.txt.gz", "bits.bin")
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        seen: set[str] = set()
        for name in bit_names:
            for bits in sorted(split_dir.glob(f"idx*/{name}")):
                sample_id = bits.parent.name
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                _add_record(records, spec, bits, split, sample_id, default_image)

    if records:
        return records

    raise FileNotFoundError(
        f"no traces under {root} "
        f"(want <root>/{{train,test}}/idx*/bits.txt)"
    )


def _label_bucket(sample: Sample) -> int:
    if sample.label.ndim == 0:
        return int(sample.label)
    bits = 0
    for i, value in enumerate(sample.label.tolist()):
        if float(value) > 0.5:
            bits |= 1 << i
    return bits


def _split_train_val_by_label(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[_label_bucket(sample)].append(idx)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(grouped):
        current = list(grouped[label])
        rng.shuffle(current)
        total = len(current)
        if total <= 1:
            train_idx.extend(current)
            continue
        raw_val = int(total * val_ratio)
        val_count = max(1, raw_val) if val_ratio > 0 else 0
        val_count = min(val_count, total - 1)
        train_idx.extend(current[: total - val_count])
        val_idx.extend(current[total - val_count :])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def _split_train_val_by_patient(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    by_patient: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        assert sample.patient_id is not None
        by_patient[int(sample.patient_id)].append(sample)
    patient_ids = list(by_patient)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    target_val = max(1, int(len(samples) * val_ratio)) if val_ratio > 0 else 0
    train: list[Sample] = []
    val: list[Sample] = []
    current_val = 0
    for patient_id in patient_ids:
        bundle = sorted(by_patient[patient_id], key=lambda item: item.sample_id)
        if current_val < target_val:
            val.extend(bundle)
            current_val += len(bundle)
        else:
            train.extend(bundle)
    if not train and val:
        train.append(val.pop())
    return train, val


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    by_split: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        if sample.split in by_split:
            by_split[sample.split].append(sample)
    if by_split["train"] and not by_split["val"]:
        if by_split["train"][0].patient_id is not None and all(
            item.patient_id is not None for item in by_split["train"]
        ):
            train, val = _split_train_val_by_patient(by_split["train"], val_ratio, seed)
        else:
            train, val = _split_train_val_by_label(by_split["train"], val_ratio, seed)
        by_split["train"] = train
        by_split["val"] = val or list(train[:1])
        return by_split
    if by_split["train"] or by_split["test"]:
        if not by_split["val"] and by_split["train"]:
            by_split["val"] = list(by_split["train"][:1])
        return by_split
    if len(samples) == 1:
        return {"train": list(samples), "val": list(samples), "test": list(samples)}
    train, val = _split_train_val_by_label(samples, val_ratio, seed)
    n_test = max(1, len(train) // 5) if train else 1
    return {
        "test": train[:n_test],
        "val": val or train[:1],
        "train": train[n_test:] or train,
    }


class TraceImageDataset(Dataset):
    def __init__(self, spec: CaseSpec, samples: list[Sample]):
        self.spec = spec
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        trace = bits_to_trace(read_bits(sample.bits_path), self.spec)
        if sample.image_path is None:
            raise FileNotFoundError(f"missing image for {sample.sample_id}")
        image01 = nchw_to_unit01(load_nchw_f32(sample.image_path, self.spec), self.spec)
        image01_t = torch.from_numpy(np.ascontiguousarray(image01))
        if self.spec.task == "multi_label":
            label = torch.from_numpy(np.asarray(sample.label, dtype=np.float32))
        else:
            label = torch.tensor(int(sample.label), dtype=torch.long)
        return trace, image01_t, label, sample.sample_id


class TraceOnlyDataset(Dataset):
    def __init__(self, spec: CaseSpec, samples: list[Sample]):
        self.spec = spec
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        trace = bits_to_trace(read_bits(sample.bits_path), self.spec)
        if self.spec.task == "multi_label":
            label = torch.from_numpy(np.asarray(sample.label, dtype=np.float32))
        else:
            label = torch.tensor(int(sample.label), dtype=torch.long)
        image01 = None
        if sample.image_path is not None and sample.image_path.is_file():
            image01 = torch.from_numpy(
                np.ascontiguousarray(
                    nchw_to_unit01(load_nchw_f32(sample.image_path, self.spec), self.spec)
                )
            )
        else:
            image01 = torch.zeros(
                (self.spec.nc, self.spec.height, self.spec.width), dtype=torch.float32
            )
        return trace, image01, label, sample.sample_id


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
