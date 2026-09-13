from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attack.common.cases import CaseSpec
from attack.common.glow import GlowBatch
from attack.common.images import model_to_victim, victim_to_model
from attack.common.util import AverageMeter, write_json
from attack.recon.nets import ImageAE, ReconstructionLoss, TraceTransform


class PairwiseDistance:
    def apply(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        eps = 1e-4 / max(1, x1.shape[1])
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, 2.0).sum(dim=1)
        return torch.pow(out + eps, 0.5)


def _binary_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_b = pred > 0.5
    target_b = target > 0.5
    inter = (pred_b & target_b).float().sum(dim=1)
    union = (pred_b | target_b).float().sum(dim=1).clamp(min=1.0)
    return inter / union


def _binary_f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_b = pred > 0.5
    target_b = target > 0.5
    tp = (pred_b & target_b).float().sum(dim=1)
    fp = (pred_b & ~target_b).float().sum(dim=1)
    fn = (~pred_b & target_b).float().sum(dim=1)
    return (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1e-8)


def semantic_checkpoint_key(metrics: dict[str, float], spec: CaseSpec) -> tuple[float, float, float, float]:
    lead = metrics.get("label_f1" if spec.task == "multi_label" else "label_acc", float("-inf"))
    return (
        float(lead),
        float(metrics.get("victim_consistency", float("-inf"))),
        float(metrics.get("victim_query_acc", float("-inf"))),
        -float(metrics.get("total", float("inf"))),
    )


def _glow_semantic(
    spec: CaseSpec,
    recon_model: torch.Tensor,
    image01: torch.Tensor,
    labels: torch.Tensor,
    glow: GlowBatch,
    device: torch.device,
) -> dict[str, float]:
    recon01 = model_to_victim(recon_model).detach().cpu().numpy()
    target01 = image01.detach().cpu().numpy()
    recon_logits = torch.from_numpy(glow.logits_batch(recon01)).to(device)
    target_logits = torch.from_numpy(glow.logits_batch(target01)).to(device)
    distance = PairwiseDistance()
    if spec.task == "multi_label":
        recon_mask = torch.sigmoid(recon_logits) >= 0.5
        target_mask = torch.sigmoid(target_logits) >= 0.5
        gt_mask = labels.to(device).float() >= 0.5
        consistency = float(_binary_iou(recon_mask.float(), target_mask.float()).mean().item())
        label_f1 = float(_binary_f1(recon_mask.float(), gt_mask.float()).mean().item())
        label_acc = label_f1
    else:
        recon_pred = recon_logits.argmax(dim=1)
        target_pred = target_logits.argmax(dim=1)
        gt = labels.to(device).long()
        consistency = float((recon_pred == target_pred).float().mean().item())
        label_acc = float((recon_pred == gt).float().mean().item())
        label_f1 = label_acc
    if recon_logits.shape[0] <= 1:
        query = 1.0
    else:
        shuffle = torch.randperm(recon_logits.shape[0], device=device)
        dist_true = distance.apply(recon_logits, target_logits)
        dist_false = distance.apply(recon_logits, target_logits[shuffle])
        query = float((dist_true < dist_false).float().mean().item())
    return {
        "label_acc": label_acc,
        "label_f1": label_f1,
        "victim_consistency": consistency,
        "victim_query_acc": query,
    }


def train_tonly(
    spec: CaseSpec,
    cipher,
    loaders: dict[str, DataLoader],
    out_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    beta1: float,
    glow: GlowBatch,
) -> Path:
    stage = out_dir / "tonly"
    ckpt_dir = stage / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = TraceTransform(cipher, spec).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    loss_fn = ReconstructionLoss(spec.loss_kind, device)
    best_path = ckpt_dir / "best.pth"
    history: list[dict] = []
    best_key = None

    def run_epoch(loader: DataLoader, training: bool) -> dict[str, float]:
        model.train(training)
        meters = {k: AverageMeter() for k in ("total", "label_acc", "label_f1", "victim_consistency", "victim_query_acc")}
        for trace, image01, labels, _sid in loader:
            trace = trace.to(device)
            image01 = image01.to(device)
            image_model = victim_to_model(image01)
            with torch.set_grad_enabled(training):
                recon = model(trace)
                loss = loss_fn(recon, image_model)
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    optim.step()
            n = int(trace.shape[0])
            meters["total"].update(float(loss.item()), n)
            if not training:
                sem = _glow_semantic(spec, recon, image01, labels, glow, device)
                for key, value in sem.items():
                    meters[key].update(value, n)
        return {k: m.avg for k, m in meters.items()}

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(loaders["train"], True)
        val_m = run_epoch(loaders["val"], False)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(
            f"[tonly] epoch={epoch:03d} train={train_m['total']:.6f} val={val_m['total']:.6f} "
            f"val_acc={val_m['label_acc']:.4f} val_consistency={val_m['victim_consistency']:.4f}",
            flush=True,
        )
        current = semantic_checkpoint_key(val_m, spec)
        if best_key is None or current > best_key:
            best_key = current
            torch.save({"epoch": epoch, "t_model": model.state_dict(), "val": val_m}, best_path)

    torch.save({"epoch": epochs, "t_model": model.state_dict()}, ckpt_dir / "final.pth")
    write_json(
        stage / "train_summary.json",
        {"best_semantic_key": list(best_key) if best_key else None, "best_ckpt": str(best_path)},
    )
    write_json(stage / "epoch_history.json", {"history": history})
    return best_path


def train_ti(
    spec: CaseSpec,
    cipher,
    loaders: dict[str, DataLoader],
    out_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    beta1: float,
    glow: GlowBatch,
    t_weight: float = 1.0,
    i_weight: float = 1.0,
) -> Path:
    stage = out_dir / "full"
    ckpt_dir = stage / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_model = TraceTransform(cipher, spec).to(device)
    i_model = ImageAE(cipher, spec).to(device)
    optim = torch.optim.Adam(
        list(t_model.parameters()) + list(i_model.parameters()),
        lr=lr,
        betas=(beta1, 0.999),
    )
    loss_fn = ReconstructionLoss(spec.loss_kind, device)
    best_path = ckpt_dir / "best.pth"
    history: list[dict] = []
    best_key = None

    def run_epoch(loader: DataLoader, training: bool) -> dict[str, float]:
        t_model.train(training)
        i_model.train(training)
        meters = {
            k: AverageMeter()
            for k in ("total", "t_loss", "i_loss", "label_acc", "label_f1", "victim_consistency", "victim_query_acc")
        }
        for trace, image01, labels, _sid in loader:
            trace = trace.to(device)
            image01 = image01.to(device)
            image_model = victim_to_model(image01)
            with torch.set_grad_enabled(training):
                decoded = t_model(trace)
                image_hat = i_model(image_model)
                t_loss = loss_fn(decoded, image_model)
                i_loss = loss_fn(image_hat, decoded)
                total = t_weight * t_loss + i_weight * i_loss
                if training:
                    optim.zero_grad(set_to_none=True)
                    total.backward()
                    optim.step()
            n = int(trace.shape[0])
            meters["total"].update(float(total.item()), n)
            meters["t_loss"].update(float(t_loss.item()), n)
            meters["i_loss"].update(float(i_loss.item()), n)
            if not training:
                sem = _glow_semantic(spec, decoded, image01, labels, glow, device)
                for key, value in sem.items():
                    meters[key].update(value, n)
        return {k: m.avg for k, m in meters.items()}

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(loaders["train"], True)
        val_m = run_epoch(loaders["val"], False)
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(
            f"[full-ti] epoch={epoch:03d} train={train_m['total']:.6f} val={val_m['total']:.6f} "
            f"val_acc={val_m['label_acc']:.4f} val_consistency={val_m['victim_consistency']:.4f}",
            flush=True,
        )
        current = semantic_checkpoint_key(val_m, spec)
        if best_key is None or current > best_key:
            best_key = current
            torch.save(
                {
                    "epoch": epoch,
                    "t_model": t_model.state_dict(),
                    "i_model": i_model.state_dict(),
                    "val": val_m,
                },
                best_path,
            )

    torch.save(
        {"epoch": epochs, "t_model": t_model.state_dict(), "i_model": i_model.state_dict()},
        ckpt_dir / "final.pth",
    )
    write_json(
        stage / "train_summary.json",
        {"best_semantic_key": list(best_key) if best_key else None, "best_ckpt": str(best_path)},
    )
    write_json(stage / "epoch_history.json", {"history": history})
    return best_path
