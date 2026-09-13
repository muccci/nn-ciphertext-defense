from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.common.cases import CaseSpec
from attack.common.glow import GlowBatch
from attack.common.images import model_to_victim
from attack.recon.prior import ChestPGGANPrior, StyleGANXLPrior


def invert_latent(
    h_target: torch.Tensor,
    i_model: nn.Module,
    g_model: nn.Module,
    spec: CaseSpec,
    device: torch.device,
    *,
    glow: GlowBatch | None = None,
) -> torch.Tensor:
    """x* so I(G(*)) ≈ h = T(trace)."""
    if spec.case == "MC01":
        return _invert_uncond_z(h_target, i_model, g_model, spec, device)
    if spec.case == "MC10":
        assert isinstance(g_model, ChestPGGANPrior)
        return _invert_pggan(h_target, i_model, g_model, spec, device)
    if spec.case in {"MC04", "MC15"}:
        assert isinstance(g_model, StyleGANXLPrior)
        return _invert_stylegan(h_target, i_model, g_model, spec, device, glow=glow)
    raise SystemExit(f"no invert for {spec.case}")


def _match_and_resize(fake: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
    if fake.shape[-2:] != h_target.shape[-2:]:
        return F.interpolate(fake, size=h_target.shape[-2:], mode="bilinear", align_corners=False)
    return fake


def _invert_uncond_z(
    h_target: torch.Tensor,
    i_model: nn.Module,
    g_model: nn.Module,
    spec: CaseSpec,
    device: torch.device,
) -> torch.Tensor:
    batch = int(h_target.shape[0])
    latent = int(g_model.latent_dim)
    best_loss = torch.full((batch,), float("inf"), device=device)
    best_fake = torch.zeros_like(h_target)
    i_model.eval()
    g_model.eval()
    for _ in range(spec.z_restarts):
        z = torch.randn((batch, latent), device=device, requires_grad=True)
        optim = torch.optim.Adam([z], lr=spec.z_lr)
        for _ in range(spec.z_steps):
            fake = _match_and_resize(g_model(z), h_target)
            pred = i_model(fake)
            match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
            reg = z.pow(2).mean(dim=1)
            objective = match + spec.z_reg_weight * reg
            optim.zero_grad(set_to_none=True)
            objective.mean().backward()
            optim.step()
        with torch.no_grad():
            fake = _match_and_resize(g_model(z), h_target)
            pred = i_model(fake)
            match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
            reg = z.pow(2).mean(dim=1)
            objective = match + spec.z_reg_weight * reg
            better = objective < best_loss
            best_loss = torch.where(better, objective, best_loss)
            best_fake = torch.where(better.view(-1, 1, 1, 1), fake, best_fake)
    return best_fake


def _invert_pggan(
    h_target: torch.Tensor,
    i_model: nn.Module,
    g_model: ChestPGGANPrior,
    spec: CaseSpec,
    device: torch.device,
) -> torch.Tensor:
    batch = int(h_target.shape[0])
    latent = int(g_model.latent_dim)
    best_loss = torch.full((batch,), float("inf"), device=device)
    best_fake = torch.zeros_like(h_target)
    i_model.eval()
    g_model.eval()
    for _ in range(spec.z_restarts):
        z = torch.randn((batch, latent), device=device)
        if spec.z_init_std > 0:
            z = z + torch.randn_like(z) * spec.z_init_std
        z = z.requires_grad_(True)
        optim = torch.optim.Adam([z], lr=spec.z_lr)
        for _ in range(spec.z_steps):
            fake = _match_and_resize(g_model(z), h_target)
            pred = i_model(fake)
            match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
            reg = z.pow(2).mean(dim=1)
            objective = match + spec.z_reg_weight * reg
            optim.zero_grad(set_to_none=True)
            objective.mean().backward()
            optim.step()
        with torch.no_grad():
            fake = _match_and_resize(g_model(z), h_target)
            pred = i_model(fake)
            match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
            reg = z.pow(2).mean(dim=1)
            objective = match + spec.z_reg_weight * reg
            better = objective < best_loss
            best_loss = torch.where(better, objective, best_loss)
            best_fake = torch.where(better.view(-1, 1, 1, 1), fake, best_fake)
    return best_fake


def _candidate_classes(
    h_target: torch.Tensor,
    spec: CaseSpec,
    g_model: StyleGANXLPrior,
    glow: GlowBatch | None,
    device: torch.device,
) -> torch.Tensor:
    batch = int(h_target.shape[0])
    if spec.stylegan_candidate_source == "all_mapped":
        return g_model.mapped_class_pool(batch, device)
    if glow is None:
        raise FileNotFoundError("MC04 invert needs Glow for victim top-k classes")
    recon01 = model_to_victim(h_target).detach().cpu().numpy()
    logits = torch.from_numpy(glow.logits_batch(recon01)).to(device)
    topk = min(max(1, spec.stylegan_candidate_topk), int(logits.shape[1]))
    return torch.topk(logits, k=topk, dim=1).indices


def _invert_stylegan(
    h_target: torch.Tensor,
    i_model: nn.Module,
    g_model: StyleGANXLPrior,
    spec: CaseSpec,
    device: torch.device,
    *,
    glow: GlowBatch | None,
) -> torch.Tensor:
    batch = int(h_target.shape[0])
    candidates = _candidate_classes(h_target, spec, g_model, glow, device)
    best_loss = torch.full((batch,), float("inf"), device=device)
    best_fake = torch.zeros_like(h_target)
    i_model.eval()
    g_model.eval()
    for slot in range(int(candidates.shape[1])):
        class_indices = candidates[:, slot].to(device=device, dtype=torch.long)
        w_center = g_model.class_w_avg(class_indices).detach()
        for _ in range(max(1, spec.stylegan_restarts_per_class)):
            if spec.latent_space == "w":
                w = (w_center + torch.randn_like(w_center) * spec.stylegan_w_init_std).requires_grad_(True)
                optim = torch.optim.Adam([w], lr=spec.z_lr)
            else:
                z_base = torch.randn((batch, g_model.z_dim), device=device)
                z_delta = torch.zeros_like(z_base, requires_grad=True)
                optim = torch.optim.Adam([z_delta], lr=spec.z_lr)
            for _ in range(spec.z_steps):
                if spec.latent_space == "w":
                    fake = _match_and_resize(g_model.synthesize_from_w(w), h_target)
                    reg = (w - w_center).pow(2).mean(dim=1)
                else:
                    z = z_base + z_delta
                    fake = _match_and_resize(
                        g_model.synthesize_from_z(z, class_indices, truncation_psi=1.0),
                        h_target,
                    )
                    reg = z_delta.pow(2).mean(dim=1)
                pred = i_model(fake)
                match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
                objective = match + spec.z_reg_weight * reg
                optim.zero_grad(set_to_none=True)
                objective.mean().backward()
                optim.step()
            with torch.no_grad():
                if spec.latent_space == "w":
                    fake = _match_and_resize(g_model.synthesize_from_w(w), h_target)
                    reg = (w - w_center).pow(2).mean(dim=1)
                else:
                    z = z_base + z_delta
                    fake = _match_and_resize(
                        g_model.synthesize_from_z(z, class_indices, truncation_psi=1.0),
                        h_target,
                    )
                    reg = z_delta.pow(2).mean(dim=1)
                pred = i_model(fake)
                match = F.mse_loss(pred, h_target, reduction="none").mean(dim=(1, 2, 3))
                objective = match + spec.z_reg_weight * reg
                better = objective < best_loss
                best_loss = torch.where(better, objective, best_loss)
                best_fake = torch.where(better.view(-1, 1, 1, 1), fake, best_fake)
    return best_fake
