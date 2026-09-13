#!/usr/bin/env python3
"""Label metrics via Glow teacher.

Usage:
  python3 attack/label/compute.py --case MC01
  python3 attack/label/compute.py --case MC15 --mode off --site transition2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parents[2]
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))

from attack.common.cases import SITES, default_site, get_spec
from attack.common.dataset import TraceOnlyDataset, discover_samples, make_loader, split_samples
from attack.common.glow import GlowBatch
from attack.common.images import load_nchw_f32, nchw_to_unit01
from attack.common.util import ART_ROOT, load_cipher_model, resolve_device, write_json
from attack.label.nets import TraceLabel, hard_teacher_loss, teacher_hard_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("MC01", "MC04", "MC10", "MC15"))
    parser.add_argument("--mode", default="", choices=("", "off", "on"))
    parser.add_argument("--site", default="")
    parser.add_argument("--traces", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    modes = [args.mode] if args.mode else ["off", "on"]
    if args.site:
        sites = [args.site]
    elif args.case == "MC15":
        sites = list(SITES["MC15"])
    else:
        sites = [default_site(args.case)]
    return [(m, s) for m in modes for s in sites]


def _macro_bacc(target: np.ndarray, pred: np.ndarray) -> float:
    scores: list[float] = []
    for col in range(target.shape[1]):
        t = target[:, col] > 0.5
        p = pred[:, col] > 0.5
        if np.unique(t.astype(np.int8)).size < 2:
            continue
        pos, neg = t, ~t
        tpr = float(p[pos].mean()) if pos.any() else float("nan")
        tnr = float((~p[neg]).mean()) if neg.any() else float("nan")
        if np.isnan(tpr) or np.isnan(tnr):
            continue
        scores.append(0.5 * (tpr + tnr))
    return float(np.mean(scores)) if scores else float("nan")


def compute_one(args: argparse.Namespace, mode: str, site: str) -> Path:
    spec = get_spec(args.case, site, mode, task="label")
    device = resolve_device(args.device)
    out = args.out or (ART_ROOT / "attack_out" / "label" / spec.case / mode / site)
    ckpt = args.ckpt or (out / "hard" / "ckpt" / "best.pth")
    if not ckpt.is_file():
        alt = ckpt.with_name("final.pth")
        ckpt = alt if alt.is_file() else ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing label ckpt {ckpt}")
    samples = discover_samples(spec, mode, args.traces)
    splits = split_samples(samples, 0.1, 20260418)
    chosen = splits.get(args.split) or splits["test"] or samples
    if args.max_items > 0:
        chosen = chosen[: args.max_items]
    loader = make_loader(
        TraceOnlyDataset(spec, chosen),
        batch_size=args.batch_size or spec.batch_size,
        shuffle=False,
        num_workers=0,
        device=device,
    )
    model = TraceLabel(load_cipher_model(), spec).to(device)
    payload = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()

    rows: list[dict] = []
    with GlowBatch(ART_ROOT, spec, mode, out / "glow_work") as glow:
        for trace, image01, labels, sids in loader:
            trace = trace.to(device)
            image01_np = image01.numpy()
            teacher = []
            for i in range(trace.shape[0]):
                teacher.append(glow.logits(image01_np[i]))
            teacher_t = torch.from_numpy(np.stack(teacher, axis=0)).to(device)
            with torch.no_grad():
                student = model(trace)
                loss = hard_teacher_loss(student, teacher_t, spec)
            if spec.task == "multi_label":
                student_mask = teacher_hard_mask(student)
                teacher_mask = teacher_hard_mask(teacher_t)
                gt_mask = labels.to(device).float()
                for i, sid in enumerate(sids):
                    rows.append(
                        {
                            "sample_id": sid,
                            "gt_pos": " ".join(
                                str(j) for j, v in enumerate(gt_mask[i].tolist()) if v > 0.5
                            ),
                            "student_pos": " ".join(
                                str(j) for j, v in enumerate(student_mask[i].tolist()) if v > 0.5
                            ),
                            "teacher_pos": " ".join(
                                str(j) for j, v in enumerate(teacher_mask[i].tolist()) if v > 0.5
                            ),
                            "teacher_bce": float(loss[i].item()),
                            "_student_mask": student_mask[i].detach().cpu().numpy(),
                            "_teacher_mask": teacher_mask[i].detach().cpu().numpy(),
                            "_gt_mask": gt_mask[i].detach().cpu().numpy(),
                        }
                    )
            else:
                pred = student.argmax(dim=1)
                teacher_pred = teacher_t.argmax(dim=1)
                for i, sid in enumerate(sids):
                    gt = int(labels[i].item())
                    rows.append(
                        {
                            "sample_id": sid,
                            "gt": gt,
                            "student_pred": int(pred[i].item()),
                            "teacher_pred": int(teacher_pred[i].item()),
                            "label_acc": int(int(pred[i].item()) == gt),
                            "teacher_consistency": int(
                                int(pred[i].item()) == int(teacher_pred[i].item())
                            ),
                            "teacher_ce": float(loss[i].item()),
                        }
                    )

    tsv = out / f"{args.split}_metrics.tsv"
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    if public_rows:
        with tsv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(public_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(public_rows)
    summary = {
        "case": spec.case,
        "mode": mode,
        "site": site,
        "n": len(rows),
    }
    if spec.task == "multi_label" and rows:
        summary["teacher_bacc"] = _macro_bacc(
            np.stack([r["_teacher_mask"] for r in rows], axis=0),
            np.stack([r["_student_mask"] for r in rows], axis=0),
        )
        summary["gt_bacc"] = _macro_bacc(
            np.stack([r["_gt_mask"] for r in rows], axis=0),
            np.stack([r["_student_mask"] for r in rows], axis=0),
        )
    elif rows:
        summary["label_acc"] = float(np.mean([r["label_acc"] for r in rows]))
        summary["teacher_consistency"] = float(np.mean([r["teacher_consistency"] for r in rows]))
    write_json(out / f"{args.split}_summary.json", summary)
    print(summary, flush=True)
    return tsv


def main() -> int:
    args = parse_args()
    for mode, site in _jobs(args):
        try:
            compute_one(args, mode, site)
        except FileNotFoundError as exc:
            print(f"missing {args.case}/{mode}/{site}: {exc}", flush=True)
            if args.mode and args.site:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
