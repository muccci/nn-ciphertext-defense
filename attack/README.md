# attack

Training and recovery only. Site-bit traces are collected by
`scripts/trace/collect_attack.sh` into `trace_out/attack/`.

`off` / `on` is the same as `victims/glow/libs`. One script covers all
cases; the case is a flag, not a filename.

| dir | attack |
|---|---|
| `recon/` | trace → image (`tonly`=\(h\), `full`=\(x^*\)) |
| `label/` | trace → label (MC01/MC04/MC10/MC15) |

Nets, invert, checkpoint, and val split are the knobs in
`attack/common/cases.py` (encoder / decoder / head / lr / epochs /
loss / recover).

- MC10 collect takes the 14108672-bit 16B-block prefix, then
  `fold_odd` (second bit of each pair) to 7054336 / `431×128×128`.
  \(I\), PGGAN invert, and label head are unchanged.
- MC01 OFF and ON recon both use all 335400 bits.
- No soft-label distill. Hard teacher only.
- Teacher and recon query use artifact Glow, not a PyTorch `.pt`.

Everything needed to train lives under `attack/` (vendored
`common/model.py` and `common/lpips/`).

## Cases

Same IDs as `victims/models/MANIFEST.txt`. Not MC05 (SqueezeNet ImageNet).

| dataset | `--case` | `--site` | trace in | fold |
|---|---|---|---|---|
| MNIST | MC01 | `fc_110` | 335400 | dense |
| CIFAR | MC04 | `conv_960` | 510464 | `125×64×64` |
| Chest | MC10 | `layer1_0_conv2` | 7054336 | `431×128×128` |
| ImageNet | MC15 | `transition1` `transition2` `transition3` | 4718592 | `288×128×128` |

Collect uses the same `taintblock16trace` as ρ, plus `-site-ip` /
`-site-bits-only` / `-site-bits-txt`. Do not post-filter `trace_out/rho_trans`.
ON is the ON runner and the same ON env as `collect_rho_trans.sh`.

MC10 writes `fold_odd` of the `layer1.0.conv2` 16B-block prefix
(14108672 → 7054336). Each pair is the odd (second) bit: 00→0, 01→1,
10→0, 11→1. Not OR of the pair.
MC15 `--site` is one of `transition1` / `transition2` / `transition3`; collect
writes that string.

## Traces

`{train,test}/idx*` is the sample's position in the dataset. Collect always
writes that tree, including the bundled one-sample run:

```
trace_out/attack/<case>/<mode>/<site>/{train,test}/idxXXXXXX/bits.txt
trace_out/attack/<case>/<mode>/<site>/{train,test}/idxXXXXXX/input_nchw_f32.bin
trace_out/attack/<case>/<mode>/<site>/{train,test}/idxXXXXXX/meta.json
```

Same `{train,test}/idx*` convention as `trace_out/rho_trans/`. Override with
`--traces`. `bits.txt` is ASCII `01` (packed `.bin` / `.gz` also accepted).
`meta.json` holds `label` / `label_index` / `gt_label`, or
`gt_positive_indices` for MC10. Collect writes `split` / `sample` /
`manifest_input`; add the label fields before training.

Teacher labels for the label attack are Glow logits on the original image,
not an extra PyTorch `.pt` victim.

## Recon / label accuracy

Query the reconstructed image or the predicted label with the artifact
Glow runner (`victims/glow/libs`, same path as
`scripts/compute_victim_acc.py`). No extra PyTorch `.pt` victim.

`tonly` trains \(T\) only (trace → image \(h\)). `full` trains \(T+I\),
then **invert the case GAN** so \(I(G(\cdot))\approx h\) and writes
\(x^*=G(\cdot)\). Collect is unchanged: every case still writes the full
bit string. Invert is recover, not training.

All four cases invert a GAN. The GAN is different:

- MC01: MNIST DCGAN. Search noise \(z\) (200 steps, 4 restarts).
  OFF and ON recon both use all 335400 bits. OFF: L1, batch 64, 80
  epochs. ON: MSE, batch 32, 50 epochs. Label also uses all bits.
- MC04: CIFAR StyleGAN-XL. Search \(w\), Glow top-2 classes, 100 steps.
- MC10: Chest PGGAN. Search \(z\) (`z_init_std=0.25`), map output with
  `soft_tanh`. Fold into T is `431×128×128` after collect
  `fold_odd`.
- MC15: ImageNet StyleGAN-XL. Conditional \(z\) from
  `priors/MC15/class_map.json` (`all_mapped`), 80 steps.

Val split is stratified by GT label (Chest by `patient_id` when meta has
it). `best.pth` ranks by Glow label acc / F1, then
consistency, then query-acc, then −loss.

Priors are under `attack/priors/`.
`--gan-ckpt` / `--gan-src` only if you override those.

## Layout

```
scripts/trace/
  collect_rho_trans.sh        # → trace_out/rho_trans/
  collect_attack.sh           # → trace_out/attack/
attack/
  common/model.py             # vendored nets
  common/lpips/               # vendored
  priors/                     # MC01 DCGAN, MC04/MC15 StyleGAN-XL, MC10 PGGAN
  recon/
  label/
attack_out/
  recon/<case>/<mode>/<tonly|full>/
  label/<case>/<mode>/<site>/
```

One-command label attack (collect → train → metrics):

```bash
./scripts/run_label_attack.sh
./scripts/run_label_attack.sh --foreground --case MC01
```

Per-sample collect / train / compute remain:

```bash
bash scripts/trace/collect_attack.sh --case MC01 --mode off --site fc_110
bash scripts/trace/collect_attack.sh --case MC15 --mode off --site transition1
python3 attack/recon/run.py --case MC01 --mode off --variant tonly
python3 attack/recon/run.py --case MC04 --mode on --variant full
python3 attack/label/run.py --case MC01 --mode off
python3 attack/label/run.py --case MC15 --mode off --site transition2
python3 attack/recon/compute.py --case MC01
python3 attack/label/compute.py --case MC01
python3 attack/label/compute.py --case MC15
```
