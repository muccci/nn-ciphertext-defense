# nn-ciphertext-defense

ONNX victims, compile scripts, Pin tool source, and attack / metric doors.

| path | role |
|---|---|
| `environment` | host packages |
| `scripts/` | collect, acc, overhead, ρ, label attack, writeback build |
| `victims/` | ONNX + Glow / TVM compile |
| `attack/` | recon / label (prior weights are not in this repo) |
| `pintrace/` | `taintblock16trace` source |
| `writeback-protect/` | Glow / LLVM / TVM submodules |

```bash
git clone --recurse-submodules https://github.com/muccci/nn-ciphertext-defense.git
bash scripts/build_writeback_protect.sh
bash victims/glow/compile.sh
bash victims/tvm/compile.sh
./scripts/run_label_attack.sh --foreground --case MC01
```
