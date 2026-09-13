from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from attack.common.cases import CaseSpec
from attack.common.util import PRIORS_DIR


class MMagicDCGANBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MMagicDCGANOutput(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        self.activate = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.conv(x))


class MNISTDCGAN(nn.Module):
    """MNIST DCGAN (MMagic, latent 100, 64x64)."""

    def __init__(self, checkpoint: Path, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.noise2feat = MMagicDCGANBlock(self.latent_dim, 1024, kernel_size=4, stride=1, padding=0)
        self.upsampling = nn.ModuleList(
            [
                MMagicDCGANBlock(1024, 512, kernel_size=4, stride=2, padding=1),
                MMagicDCGANBlock(512, 256, kernel_size=4, stride=2, padding=1),
                MMagicDCGANBlock(256, 128, kernel_size=4, stride=2, padding=1),
            ]
        )
        self.output_layer = MMagicDCGANOutput(128, 1, kernel_size=4, stride=2, padding=1)
        raw = torch.load(checkpoint, map_location="cpu")
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        gen = {k[len("generator.") :]: v for k, v in state.items() if str(k).startswith("generator.")}
        if not gen:
            raise KeyError(f"no generator.* in {checkpoint}")
        self.load_state_dict(gen, strict=True)
        self.eval()
        self.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.view(z.shape[0], z.shape[1], 1, 1)
        out = self.noise2feat(z.float())
        for block in self.upsampling:
            out = block(out)
        return self.output_layer(out)


def default_prior_paths(spec: CaseSpec) -> tuple[Path, Path | None]:
    if spec.case == "MC01":
        return PRIORS_DIR / "MC01" / "dcgan.pth", None
    if spec.case == "MC04":
        return PRIORS_DIR / "MC04" / "stylegan_xl.pkl", PRIORS_DIR / "MC04" / "stylegan_xl"
    if spec.case == "MC10":
        return PRIORS_DIR / "MC10" / "pggan.pth", PRIORS_DIR / "MC10"
    if spec.case == "MC15":
        return PRIORS_DIR / "MC15" / "stylegan_xl.pkl", PRIORS_DIR / "MC15" / "stylegan_xl"
    raise SystemExit(f"unsupported prior case {spec.case}")


def load_prior(spec: CaseSpec, gan_ckpt: Path | None, gan_src: Path | None) -> nn.Module:
    default_ckpt, default_src = default_prior_paths(spec)
    ckpt = Path(gan_ckpt) if gan_ckpt is not None else default_ckpt
    src = Path(gan_src) if gan_src is not None else default_src
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing {spec.case} prior {ckpt}")
    if spec.gan_kind == "mnist_dcgan":
        return MNISTDCGAN(ckpt)
    if spec.gan_kind == "stylegan_xl":
        return StyleGANXLPrior(spec, ckpt, src)
    if spec.gan_kind == "pggan":
        return ChestPGGANPrior(spec, ckpt, src)
    raise SystemExit(f"unsupported gan {spec.gan_kind}")


def _to_model_size(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (height, width):
        return image
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)


class StyleGANXLPrior(nn.Module):
    """StyleGAN-XL wrapper: class-conditional w/z synthesize."""

    def __init__(self, spec: CaseSpec, ckpt: Path, gan_src: Path | None):
        super().__init__()
        if gan_src is None or not Path(gan_src).is_dir():
            raise FileNotFoundError("--gan-src must point at the StyleGAN-XL source tree")
        import sys

        src = str(Path(gan_src))
        if src not in sys.path:
            sys.path.insert(0, src)
        import dnnlib  # type: ignore
        import legacy  # type: ignore

        with dnnlib.util.open_url(str(ckpt)) as handle:
            network = legacy.load_network_pkl(handle)["G_ema"]
        network.eval()
        network.requires_grad_(False)
        self.generator = network
        self.height = spec.height
        self.width = spec.width
        self.c_dim = int(getattr(network, "c_dim", 0))
        self.z_dim = int(getattr(network, "z_dim", 64))
        self.w_dim = int(network.mapping.w_avg.shape[-1])
        self.num_ws = int(network.mapping.num_ws)
        self.latent_dim = self.w_dim if spec.latent_space == "w" else self.z_dim
        self.class_map, self.class_candidates = _load_class_map(spec)

    def class_onehot(self, class_indices: torch.Tensor) -> torch.Tensor:
        return F.one_hot(class_indices.to(dtype=torch.long), num_classes=self.c_dim).float()

    def class_w_avg(self, class_indices: torch.Tensor) -> torch.Tensor:
        indices = class_indices.to(device=self.generator.mapping.w_avg.device, dtype=torch.long)
        return self.generator.mapping.w_avg.index_select(0, indices)

    def synthesize_from_w(self, w: torch.Tensor) -> torch.Tensor:
        ws = w.unsqueeze(1).expand(-1, self.num_ws, -1)
        raw = self.generator.synthesis(ws, noise_mode="const")
        return _to_model_size(raw, self.height, self.width)

    def synthesize_from_z(
        self,
        z: torch.Tensor,
        class_indices: torch.Tensor,
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        labels = self.class_onehot(class_indices).to(device=z.device, dtype=z.dtype)
        raw = self.generator(z, labels, truncation_psi=float(truncation_psi), noise_mode="const")
        return _to_model_size(raw, self.height, self.width)

    def mapped_class_pool(self, batch: int, device: torch.device) -> torch.Tensor:
        merged: list[int] = []
        for row in self.class_candidates:
            for item in row:
                if int(item) not in merged:
                    merged.append(int(item))
        if not merged:
            merged = list(self.class_map)
        if not merged:
            raise ValueError("MC15 class_map.json has no StyleGAN classes")
        return torch.tensor(merged, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("use synthesize_from_w / synthesize_from_z")


def _load_class_map(spec: CaseSpec) -> tuple[list[int], list[list[int]]]:
    if spec.case != "MC15":
        return [], []
    path = PRIORS_DIR / "MC15" / "class_map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = [int(x) for x in payload.get("stylegan_primary_class_indices", [])]
    candidates = payload.get("stylegan_candidate_class_indices", [])
    if not isinstance(candidates, list) or len(candidates) != len(primary):
        candidates = [[value] for value in primary]
    rows = []
    for row in candidates:
        if not isinstance(row, list) or not row:
            raise ValueError(f"bad class_map candidate row in {path}")
        rows.append([int(item) for item in row])
    return primary, rows


class ChestPGGANPrior(nn.Module):
    """PGGAN prior. soft_tanh gray, then [-1,1] at 224."""

    def __init__(self, spec: CaseSpec, ckpt: Path, gan_src: Path | None):
        super().__init__()
        import importlib
        import sys

        pkg = Path(gan_src) if gan_src is not None else ckpt.parent
        if not pkg.is_dir():
            raise FileNotFoundError(f"PGGAN package missing: {pkg}")
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))
        network_module = importlib.import_module("model20.network")
        config_module = importlib.import_module("model20.configuration")
        progan = network_module.ProGAN(config_module.hparams).load_model(
            str(ckpt),
            map_location=torch.device("cpu"),
            image_size=256,
        )
        generator = progan.gen_shadow.eval()
        generator.requires_grad_(False)
        self.generator = generator
        self.latent_dim = 512
        self.height = spec.height
        self.width = spec.width
        self.output_mapping = spec.pggan_output_mapping or "soft_tanh"
        self.output_tau = 1.5
        self.output_eps = 0.02

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.generator(z.float())
        if raw.shape[1] == 3:
            raw = raw.mean(dim=1, keepdim=True)
        elif raw.shape[1] != 1:
            raw = raw[:, :1]
        if self.output_mapping == "soft_tanh":
            tau = max(self.output_tau, 1e-6)
            eps = min(max(self.output_eps, 0.0), 0.49)
            gray_01 = torch.sigmoid(raw / tau)
            if eps > 0.0:
                gray_01 = gray_01 * (1.0 - 2.0 * eps) + eps
        else:
            gray_01 = raw.clamp(0.0, 1.0)
        gray_01 = _to_model_size(gray_01, self.height, self.width)
        return gray_01.mul(2.0).sub(1.0)
