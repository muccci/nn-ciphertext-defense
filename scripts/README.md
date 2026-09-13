# scripts

Five raw dumps live under `data/{mnist,cifar10,imagenet50_96,celeba,chestxray14}/`.
Each job already has a fixed input door. These scripts only **fill that door** from the raw dumps. The door itself does not change.

| job | already accepts (do not change) | how raw data gets there |
|---|---|---|
| acc / overhead **MC TV TA** | runner `--stream` float32 NCHW | `compute_victim_acc.image_to_nchw` from the raw image |
| acc / overhead **IC** | `image-classifier` + `.npy` list | same `image_to_nchw`, then `np.save` (already in acc) |
| collect **MC TV TA** | `input_nchw_f32.bin` + Pin `-taint-file` | `collect_inputs.py --kind bin` |
| collect **IC** | PNG + `-image-mode` + Pin `input-tensor` | `collect_inputs.py --kind png` (same encode as the old standalone helper) |

Same `(family, variant, split, idx)` is the same original image.

| file | role |
|---|---|
| `collect_inputs.py` | raw → collect `.bin` and IC `.png` + `image_mode.txt` |
| `trace/collect_rho_trans.sh` | Pin ρ_trans |
| `trace/collect_attack.sh` | Pin attack site bits |
| `compute_victim_acc.py` | acc; owns `SPECS` + `image_to_nchw` |
| `compute_victim_overhead.py` | overhead (same tensors as acc) |
| `compute_victim_rho_trans.py` | ρ_trans from traces |
| `compute_victim_rho_static.py` | ρ_static from ONNX ReLU/Clip writes (no Pin) |
| `run_perturbation.py` | sweep those two jobs across perturbation strengths |
| `run_label_attack.py` | MC01/04/10/15 collect → label train → compute |
| `run_label_attack.sh` | same; `./scripts/run_label_attack.sh` |
| `build_writeback_protect.sh` | build writeback |

`run_perturbation.py` only schedules existing doors. `acc_out/` stays the single-config acc job. This sweep writes:

```
perturbation_out/acc/        acc.tsv         acc.png         <case>/<config>/
perturbation_out/rho_trans/  rho_trans.tsv   rho_trans.png   off/  fv_*/
```

Default 14 MC (no CelebA), fv `{106,108,110,112,114,116,120,124,126}` plus off.

```bash
python3 scripts/run_perturbation.py
python3 scripts/run_perturbation.py --stage pin --case MC02
python3 scripts/run_perturbation.py --stage acc --limit 8
python3 scripts/run_perturbation.py --stage plot
```

`compute_victim_rho_static.py` is its own door. It does not collect Pin and does
not write `acc_out/` or `trace_out/`. Default 11 MC (no CelebA, no MobileNet),
T=512, static set `{0}` only:

```
rho_static_out/  rho_static.tsv  <MC>/verdict.json
```

```bash
python3 scripts/compute_victim_rho_static.py
python3 scripts/compute_victim_rho_static.py --case MC01
```

`run_label_attack.sh` is the one-command label attack. It only calls
`collect_attack.sh` / `attack/label/run.py` / `attack/label/compute.py`.
Default four cases in the background. Prefers the statistical env for CUDA.

```bash
./scripts/run_label_attack.sh
./scripts/run_label_attack.sh --foreground --case MC01
```

Writes `trace_out/attack/` and `attack_out/label/`. Logs:
`attack_out/label/logs/<MC>.log`.

`data/inputs/` is a product, not a sixth dataset.
