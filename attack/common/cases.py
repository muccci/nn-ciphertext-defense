from __future__ import annotations

from dataclasses import dataclass, replace


SITES = {
    "MC01": ("fc_110",),
    "MC04": ("conv_960",),
    "MC10": ("layer1_0_conv2",),
    "MC15": ("transition1", "transition2", "transition3"),
}


@dataclass(frozen=True)
class CaseSpec:
    case: str
    site: str
    bits: int
    fold: tuple[int, int, int] | None
    pad_bit: str
    height: int
    width: int
    nc: int
    nz: int
    n_out: int
    task: str
    family: str
    dataset: str
    loss_kind: str
    t_epochs: int
    t_lr: float
    batch_size: int
    ti_epochs: int
    ti_lr: float
    ti_batch_size: int
    head_hidden1: int
    head_hidden2: int
    head_dropout: float
    label_epochs: int
    label_lr: float
    trace_stride: int
    latent_space: str
    stylegan_candidate_source: str
    stylegan_candidate_topk: int
    stylegan_restarts_per_class: int
    stylegan_w_init_std: float
    z_steps: int
    z_lr: float
    z_restarts: int
    z_reg_weight: float
    z_init_std: float
    pggan_output_mapping: str
    gan_kind: str


# MC10: collect fold_odd of the 14108672 prefix → 7054336 / 431×128×128.
_SPECS: dict[tuple[str, str], CaseSpec] = {}


def _add(spec: CaseSpec) -> None:
    _SPECS[(spec.case, spec.site)] = spec


_add(
    CaseSpec(
        case="MC01",
        site="fc_110",
        bits=335400,
        fold=None,
        pad_bit="0",
        height=64,
        width=64,
        nc=1,
        nz=128,
        n_out=10,
        task="single_label",
        family="lenet",
        dataset="mnist64",
        loss_kind="mse",
        t_epochs=50,
        t_lr=2e-3,
        batch_size=32,
        ti_epochs=50,
        ti_lr=2e-3,
        ti_batch_size=32,
        head_hidden1=120,
        head_hidden2=84,
        head_dropout=0.1,
        label_epochs=80,
        label_lr=2e-3,
        trace_stride=1,
        latent_space="z",
        stylegan_candidate_source="",
        stylegan_candidate_topk=0,
        stylegan_restarts_per_class=1,
        stylegan_w_init_std=0.0,
        z_steps=200,
        z_lr=3e-2,
        z_restarts=4,
        z_reg_weight=2e-4,
        z_init_std=0.0,
        pggan_output_mapping="",
        gan_kind="mnist_dcgan",
    )
)
_add(
    CaseSpec(
        case="MC04",
        site="conv_960",
        bits=510464,
        fold=(125, 64, 64),
        pad_bit="1",
        height=32,
        width=32,
        nc=3,
        nz=128,
        n_out=10,
        task="single_label",
        family="squeezenet",
        dataset="cifar32",
        loss_kind="l1_lpips",
        t_epochs=80,
        t_lr=2e-3,
        batch_size=16,
        ti_epochs=80,
        ti_lr=2e-3,
        ti_batch_size=8,
        head_hidden1=256,
        head_hidden2=128,
        head_dropout=0.1,
        label_epochs=80,
        label_lr=2e-3,
        trace_stride=1,
        latent_space="w",
        stylegan_candidate_source="victim_topk",
        stylegan_candidate_topk=2,
        stylegan_restarts_per_class=1,
        stylegan_w_init_std=5e-2,
        z_steps=100,
        z_lr=5e-2,
        z_restarts=1,
        z_reg_weight=1e-3,
        z_init_std=0.0,
        pggan_output_mapping="",
        gan_kind="stylegan_xl",
    )
)
_add(
    CaseSpec(
        case="MC10",
        site="layer1_0_conv2",
        bits=7054336,
        fold=(431, 128, 128),
        pad_bit="1",
        height=224,
        width=224,
        nc=1,
        nz=512,
        n_out=14,
        task="multi_label",
        family="resnet",
        dataset="chest_single_channel",
        loss_kind="l1_lpips",
        t_epochs=80,
        t_lr=2e-3,
        batch_size=4,
        ti_epochs=80,
        ti_lr=2e-3,
        ti_batch_size=4,
        head_hidden1=512,
        head_hidden2=256,
        head_dropout=0.2,
        label_epochs=80,
        label_lr=2e-3,
        trace_stride=1,
        latent_space="z",
        stylegan_candidate_source="",
        stylegan_candidate_topk=0,
        stylegan_restarts_per_class=1,
        stylegan_w_init_std=0.0,
        z_steps=120,
        z_lr=5e-2,
        z_restarts=4,
        z_reg_weight=1e-3,
        z_init_std=0.25,
        pggan_output_mapping="soft_tanh",
        gan_kind="pggan",
    )
)
for _site in SITES["MC15"]:
    _add(
        CaseSpec(
            case="MC15",
            site=_site,
            bits=4718592,
            fold=(288, 128, 128),
            pad_bit="1",
            height=96,
            width=96,
            nc=3,
            nz=512,
            n_out=50,
            task="single_label",
            family="densenet",
            dataset="imagenet50_96",
            loss_kind="l1_lpips",
            t_epochs=80,
            t_lr=2e-3,
            batch_size=16,
            ti_epochs=80,
            ti_lr=2e-3,
            ti_batch_size=16,
            head_hidden1=512,
            head_hidden2=256,
            head_dropout=0.2,
            label_epochs=80,
            label_lr=2e-3,
            trace_stride=1,
            latent_space="z",
            stylegan_candidate_source="all_mapped",
            stylegan_candidate_topk=3,
            stylegan_restarts_per_class=1,
            stylegan_w_init_std=5e-2,
            z_steps=80,
            z_lr=5e-2,
            z_restarts=1,
            z_reg_weight=1e-2,
            z_init_std=0.0,
            pggan_output_mapping="",
            gan_kind="stylegan_xl",
        )
    )


def default_site(case: str) -> str:
    sites = SITES.get(case)
    if not sites:
        raise SystemExit(f"unsupported case {case}")
    if len(sites) != 1:
        raise SystemExit(f"{case} needs --site {'|'.join(sites)}")
    return sites[0]


def effective_trace_len(spec: CaseSpec) -> int:
    if spec.fold is not None:
        return int(spec.fold[0] * spec.fold[1] * spec.fold[2])
    stride = max(1, int(spec.trace_stride))
    return (int(spec.bits) - 1) // stride + 1


def get_spec(
    case: str,
    site: str | None = None,
    mode: str | None = None,
    *,
    task: str = "recon",
) -> CaseSpec:
    if site is None or site == "":
        site = default_site(case)
    key = (case, site)
    if key not in _SPECS:
        raise SystemExit(f"unsupported {case} --site {site}")
    spec = _SPECS[key]
    # Label attack always uses the full bit string.
    # MC01 OFF/ON both train with stride 1 / batch 32.
    if task == "label":
        if case == "MC01":
            return replace(spec, trace_stride=1, batch_size=32)
        return spec
    if case == "MC01" and mode == "off":
        # OFF recon uses all 335400 bits (stride 1).
        return replace(
            spec,
            loss_kind="l1",
            batch_size=64,
            ti_batch_size=64,
            t_epochs=80,
            ti_epochs=80,
            trace_stride=1,
            z_steps=200,
            z_restarts=4,
        )
    if case == "MC01" and mode == "on":
        # ON recon uses all 335400 bits.
        return replace(
            spec,
            loss_kind="mse",
            batch_size=32,
            ti_batch_size=32,
            t_epochs=50,
            ti_epochs=50,
            trace_stride=1,
            z_steps=200,
            z_restarts=4,
        )
    return spec
