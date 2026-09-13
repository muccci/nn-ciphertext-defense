#!/usr/bin/env python3
"""Raw data/ dumps → files collect already takes.

MC/TV/TA: input_nchw_f32.bin
IC: input_n0.png + image_mode.txt (same encode as the old nchw_bin_to_png.py)

Does not change acc / overhead doors or IC's PNG + -image-mode CLI.

  python3 scripts/collect_inputs.py resolve --family resnet --variant celea --split test --sample idx000042
  python3 scripts/collect_inputs.py resolve --family resnet --variant celea --split test --sample idx000042 --kind png
"""
from __future__ import annotations

import argparse
import binascii
import csv
import json
import pickle
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import compute_victim_acc as acc

ART = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ART / "data"
DEFAULT_INPUTS = ART / "data" / "inputs"
MANIFEST = ART / "victims" / "models" / "MANIFEST.txt"

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE = Image.BILINEAR


def parse_sample(raw: str) -> int:
    text = str(raw).strip()
    if text.startswith("idx"):
        text = text[3:]
    if text.isdigit():
        return int(text)
    return 0


def sample_name(index: int) -> str:
    return f"idx{int(index):06d}"


def spec_for(family: str, variant: str) -> acc.DataSpec:
    key = (family.lower(), variant)
    if key not in acc.SPECS:
        raise SystemExit(f"no spec for {family}/{variant}")
    return acc.SPECS[key]


def data_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli
    if DEFAULT_DATA.is_dir():
        return DEFAULT_DATA
    return acc.resolve_dataset_root(None)


def imagenet_root(data: Path) -> Path:
    return acc.resolve_imagenet50_root(None, data)


def load_partition(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, code = line.split()
        out[name] = int(code)
    return out


def load_identity(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, ident = line.split()
        out[name] = int(ident) - 1
    return out


def celeba_paths(root: Path, split: str) -> list[Path]:
    image_root = root / "celeba" / "Dataset" / "CelebA_train" / "img_align_celeba"
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing CelebA images: {image_root}")
    files = sorted(p for p in image_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    part = load_partition(root / "celeba" / "annotations" / "list_eval_partition.txt")
    if part:
        code = {"train": 0, "val": 1, "test": 2}.get(split)
        if code is None:
            raise SystemExit(f"celeba split must be train|val|test, got {split}")
        files = [p for p in files if part.get(p.name) == code]
    if not files:
        raise FileNotFoundError(f"no CelebA files for split={split} under {image_root}")
    return files


def _read_idx(path: Path, header: int) -> tuple[tuple[int, ...], bytes]:
    raw = path.read_bytes()
    magic, *dims = struct.unpack(">" + "I" * (1 + header), raw[: 4 * (1 + header)])
    if magic not in (2049, 2051):
        raise RuntimeError(f"bad idx magic {magic} in {path}")
    return tuple(dims), raw[4 * (1 + header):]


def mnist_item(root: Path, split: str, index: int) -> tuple[Image.Image, int, dict[str, Any]]:
    raw_dir = root / "mnist" / "MNIST" / "raw"
    if split == "train":
        images_p, labels_p = raw_dir / "train-images-idx3-ubyte", raw_dir / "train-labels-idx1-ubyte"
    else:
        images_p, labels_p = raw_dir / "t10k-images-idx3-ubyte", raw_dir / "t10k-labels-idx1-ubyte"
    (n, rows, cols), pixels = _read_idx(images_p, 3)
    (n_l,), labels = _read_idx(labels_p, 1)
    if index < 0 or index >= n or n != n_l:
        raise IndexError(f"mnist {split} index {index} n={n}")
    start = index * rows * cols
    plane = pixels[start:start + rows * cols]
    image = Image.frombytes("L", (cols, rows), plane)
    return image, int(labels[index]), {"filename": f"mnist_{split}_{index}", "source_index": index}


def cifar_item(root: Path, split: str, index: int) -> tuple[Image.Image, int, dict[str, Any]]:
    batch_dir = root / "cifar10" / "cifar-10-batches-py"
    if split == "train":
        names = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        names = ["test_batch"]
    offset = 0
    for name in names:
        with (batch_dir / name).open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        data = payload[b"data"]
        labels = payload[b"labels"]
        if index < offset + len(labels):
            local = index - offset
            chw = data[local].reshape(3, 32, 32)
            hw = chw.transpose(1, 2, 0)
            image = Image.fromarray(hw, mode="RGB")
            return image, int(labels[local]), {
                "filename": f"cifar10_{split}_{index}",
                "source_index": index,
            }
        offset += len(labels)
    raise IndexError(f"cifar10 {split} index {index}")


def folder_items(split_root: Path, label_map: dict[str, Any]) -> list[tuple[Path, int]]:
    class_dirs = [str(x) for x in label_map["class_dirs"]]
    samples: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(class_dirs):
        class_root = split_root / class_dir
        if not class_root.is_dir():
            raise FileNotFoundError(class_root)
        for path in sorted(p for p in class_root.iterdir() if p.is_file()):
            samples.append((path, label))
    if not samples:
        raise RuntimeError(f"no images under {split_root}")
    return samples


def imagenet_item(root: Path, split: str, index: int) -> tuple[Image.Image, int, dict[str, Any]]:
    inet = imagenet_root(root)
    label_map = json.loads((inet / "label_map.json").read_text(encoding="utf-8"))
    split_name = "train" if split == "train" else "validation"
    path, label = folder_items(inet / split_name, label_map)[index]
    with Image.open(path) as image:
        return image.convert("RGB"), int(label), {
            "filename": path.name,
            "path": str(path),
            "source_index": index,
        }


def celeba_item(root: Path, split: str, index: int) -> tuple[Image.Image, int, dict[str, Any]]:
    path = celeba_paths(root, split)[index]
    identity = load_identity(root / "celeba" / "annotations" / "identity_CelebA.txt")
    label = identity.get(path.name, -1)
    with Image.open(path) as image:
        return image.convert("RGB"), int(label), {
            "filename": path.name,
            "path": str(path),
            "source_index": index,
        }


def chest_names(root: Path, split: str) -> list[str]:
    data_dir = root / "chestxray14"
    image_dir = data_dir / "images"
    test_list = {
        line.strip()
        for line in (data_dir / "test_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if split == "test":
        names = sorted(name for name in test_list if (image_dir / name).is_file())
    else:
        names = sorted(
            p.name for p in image_dir.iterdir()
            if p.is_file() and p.name not in test_list
        )
    if not names:
        raise FileNotFoundError(f"no ChestX-ray14 names for split={split}")
    return names


def chest_item(root: Path, split: str, index: int) -> tuple[Image.Image, Any, dict[str, Any]]:
    data_dir = root / "chestxray14"
    name = chest_names(root, split)[index]
    path = data_dir / "images" / name
    target = [0.0] * len(acc.CHEST_LABELS)
    patient_id = None
    csv_path = data_dir / "Data_Entry_2017_v2020.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["Image Index"] != name:
                    continue
                labels = row["Finding Labels"]
                if labels != "No Finding":
                    for label in labels.split("|"):
                        if label in acc.CHEST_LABELS:
                            target[acc.CHEST_LABELS.index(label)] = 1.0
                if row.get("Patient ID"):
                    patient_id = int(row["Patient ID"])
                break
    with Image.open(path) as image:
        return image.convert("L"), target, {
            "filename": name,
            "path": str(path),
            "source_index": index,
            "patient_id": patient_id,
            "gt_positive_indices": [i for i, v in enumerate(target) if v > 0],
        }


def raw_item(
    spec: acc.DataSpec, root: Path, split: str, index: int,
) -> tuple[Image.Image, Any, dict[str, Any]]:
    if spec.source == "mnist":
        return mnist_item(root, split, index)
    if spec.source == "cifar10":
        return cifar_item(root, split, index)
    if spec.source == "imagenet50_96":
        return imagenet_item(root, split, index)
    if spec.source == "celeba":
        return celeba_item(root, split, index)
    if spec.source == "chestxray14":
        return chest_item(root, split, index)
    raise SystemExit(f"unsupported source {spec.source}")


def prepared_dir(inputs: Path, family: str, variant: str, split: str, index: int) -> Path:
    return inputs / family / variant / split / sample_name(index)


def png_gray(plane: np.ndarray) -> bytes:
    hh, ww = plane.shape
    raw = b"".join(b"\x00" + plane[i].tobytes() for i in range(hh))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", binascii.crc32(tag + data) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", ww, hh, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def png_rgb(chw: np.ndarray) -> bytes:
    hh, ww = chw.shape[1], chw.shape[2]
    rows = []
    for y in range(hh):
        row = bytearray([0])
        for x in range(ww):
            row.extend((int(chw[0, y, x]), int(chw[1, y, x]), int(chw[2, y, x])))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", binascii.crc32(tag + data) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", ww, hh, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_ic_png(
    arr: np.ndarray,
    dest: Path,
    zero_one_bin: Path | None = None,
) -> Path:
    """Same encode as the former nchw_bin_to_png.py (IC PNG + image-mode)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    n, c, h, w = (int(x) for x in arr.shape)
    z1 = zero_one_bin
    if z1 is not None and z1.is_file() and c == 1 and h == 28 and w == 28:
        a01 = np.fromfile(z1, dtype=np.float32)
        if a01.size == 784 and float(a01.min()) >= 0.0 and float(a01.max()) <= 1.0 + 1e-6:
            if float(arr.min()) < -1e-3 or float(arr.max()) > 1.0 + 1e-3 or (arr == 0).mean() < 0.1:
                arr = a01.reshape(1, 1, 28, 28)
                n = 1
    amin, amax = float(arr.min()), float(arr.max())
    if amin >= -1e-6 and amax <= 1.0 + 1e-3:
        image_mode = "0to1"
        pix = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif amin >= -1e-6 and amax <= 255.0 + 1e-3:
        image_mode = "0to255"
        pix = np.clip(arr, 0, 255).astype(np.uint8)
    elif amin >= -1.0 - 1e-3 and amax <= 1.0 + 1e-3:
        image_mode = "neg1to1"
        pix = np.clip((arr + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    else:
        image_mode = "0to1"
        scaled = (arr - amin) / (amax - amin + 1e-12)
        pix = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for bi in range(n):
        if c == 1:
            data = png_gray(pix[bi, 0])
        elif c == 3:
            data = png_rgb(pix[bi])
        else:
            raise SystemExit(f"unsupported channels c={c}")
        out = dest / f"input_n{bi}.png"
        out.write_bytes(data)
        paths.append(str(out.resolve()))
    (dest / "image.list").write_text("\n".join(paths) + "\n")
    (dest / "image_mode.txt").write_text(image_mode + "\n")
    return dest / "input_n0.png"


def write_sample(
    family: str,
    variant: str,
    split: str,
    index: int,
    root: Path,
    inputs: Path,
) -> Path:
    spec = spec_for(family, variant)
    image, label, extra = raw_item(spec, root, split, index)
    tensor = acc.image_to_nchw(image, spec)
    dest = prepared_dir(inputs, family, variant, split, index)
    dest.mkdir(parents=True, exist_ok=True)
    bin_path = dest / "input_nchw_f32.bin"
    tensor.astype("float32").tofile(bin_path)
    write_ic_png(tensor, dest)
    meta = {
        "family": family,
        "variant": variant,
        "source": spec.source,
        "split": split,
        "sample": sample_name(index),
        "index": int(index),
        "shape": [int(x) for x in tensor.shape],
        "height": spec.height,
        "width": spec.width,
        "channels": spec.channels,
        "mean": list(spec.mean),
        "std": list(spec.std),
        "label": label if not hasattr(label, "tolist") else label,
        **extra,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return bin_path


def resolve_input(
    family: str,
    variant: str,
    split: str,
    sample: str,
    root: Path,
    inputs: Path,
    kind: str = "bin",
    prepare: bool = True,
    zero_one_bin: Path | None = None,
) -> Path:
    index = parse_sample(sample)
    dest = prepared_dir(inputs, family, variant, split, index)
    bin_path = dest / "input_nchw_f32.bin"
    png_path = dest / "input_n0.png"
    if kind not in {"bin", "png"}:
        raise SystemExit(f"kind must be bin|png, got {kind}")
    if kind == "bin":
        if bin_path.is_file():
            return bin_path
        if not prepare:
            raise FileNotFoundError(bin_path)
        return write_sample(family, variant, split, index, root, inputs)
    if png_path.is_file() and (dest / "image_mode.txt").is_file() and zero_one_bin is None:
        return png_path
    if not prepare:
        raise FileNotFoundError(png_path)
    if not bin_path.is_file():
        write_sample(family, variant, split, index, root, inputs)
        if zero_one_bin is None:
            return dest / "input_n0.png"
    spec = spec_for(family, variant)
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(
        1, spec.channels, spec.height, spec.width
    )
    return write_ic_png(arr, dest, zero_one_bin)


def manifest_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("MC"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        family, variant = parts[3].split("/")[:2]
        key = (family, variant)
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows


def cmd_resolve(args: argparse.Namespace) -> int:
    path = resolve_input(
        args.family, args.variant, args.split, args.sample,
        data_root(args.dataset_root), args.inputs,
        kind=args.kind, prepare=not args.no_prepare,
        zero_one_bin=args.zero_one_bin,
    )
    print(path)
    return 0


def split_n(family: str, variant: str, split: str, root: Path | None = None) -> int:
    spec = spec_for(family, variant)
    data = data_root(root)
    if spec.source == "mnist":
        raw_dir = data / "mnist" / "MNIST" / "raw"
        path = raw_dir / (
            "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
        )
        dims, _ = _read_idx(path, 3)
        return int(dims[0])
    if spec.source == "cifar10":
        return 50000 if split == "train" else 10000
    if spec.source == "imagenet50_96":
        inet = imagenet_root(data)
        label_map = json.loads((inet / "label_map.json").read_text(encoding="utf-8"))
        split_name = "train" if split == "train" else "validation"
        return len(folder_items(inet / split_name, label_map))
    if spec.source == "chestxray14":
        return len(chest_names(data, split))
    if spec.source == "celeba":
        return len(celeba_paths(data, split))
    raise SystemExit(f"unsupported source {spec.source}")


def list_indices(
    family: str,
    variant: str,
    split: str,
    limit: int = 0,
    root: Path | None = None,
) -> list[int]:
    total = split_n(family, variant, split, root)
    indexes = list(range(total))
    if limit > 0:
        indexes = indexes[: min(limit, total)]
    return indexes


def cmd_list(args: argparse.Namespace) -> int:
    indexes = list_indices(
        args.family, args.variant, args.split, args.limit, args.dataset_root
    )
    for index in indexes:
        print(sample_name(index))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    root = data_root(args.dataset_root)
    pairs = [(args.family, args.variant)] if args.family and args.variant else manifest_rows()
    if args.family and not args.variant:
        raise SystemExit("need --variant with --family")
    indexes = [int(x) for x in args.index]
    for family, variant in pairs:
        for index in indexes:
            path = write_sample(family, variant, args.split, index, root, args.inputs)
            print(f"{family}/{variant}/{args.split}/{sample_name(index)} {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="print collect bin or IC png path; write if missing")
    r.add_argument("--family", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--sample", required=True)
    r.add_argument("--kind", choices=("bin", "png"), default="bin")
    r.add_argument("--zero-one-bin", type=Path, default=None)
    r.add_argument("--dataset-root", type=Path, default=None)
    r.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    r.add_argument("--no-prepare", action="store_true")
    r.set_defaults(func=cmd_resolve)

    p = sub.add_parser("prepare", help="write bins from raw data/")
    p.add_argument("--family", default="")
    p.add_argument("--variant", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--index", nargs="+", default=["42"], help="dataset positions, default 42")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    p.set_defaults(func=cmd_prepare)

    lst = sub.add_parser("list", help="print idx* names for a split")
    lst.add_argument("--family", required=True)
    lst.add_argument("--variant", required=True)
    lst.add_argument("--split", default="test")
    lst.add_argument("--limit", type=int, default=0, help="0 = all")
    lst.add_argument("--dataset-root", type=Path, default=None)
    lst.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
