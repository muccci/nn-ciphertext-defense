# pintrace (delivery)

Minimal Pin kit + `taintblock16trace` for ρ_trans and attack site bits:

- **MC / TV / TA**: `-taint-seed-mode file` + `-taint-file <input.bin>`
- **IC**: `-taint-seed-mode input-tensor` (Glow `updateInputPlaceholders` /
  `Tensor::assign` seed; used with `image-classifier` + PNG)

## Layout

```text
pintrace/
  pin/                 Intel Pin launcher + intel64 runtime (+ extras, licensing)
  tool/
    taintblock16trace.cpp
    makefile
    makefile.rules
    obj-intel64/taintblock16trace.so   # prebuilt
  README.md
```

Not shipped (not needed for 64-bit runs): `ia32/`, `pin32`,
`bindings/`, `doc/`, other pintools (`paddrtrace`, `sitebittrace`, …).

Launch surface kept for scripts: `-o`, `-m`, `-taint-file`,
`-taint-seed-mode file|input-tensor`, ignored `-stack-depth`. Extra research
knobs from the Glow pintrace source (site/caller filters, event bins) remain
available. `scripts/trace/collect_rho_trans.sh` does not pass them;
`scripts/trace/collect_attack.sh` uses `-site-ip` / `-site-bits-only`.

## Run MC (file seed)

From the `artifact/` root:

```bash
PIN=pintrace/pin/pin
TOOL=pintrace/tool/obj-intel64/taintblock16trace.so
RUNNER=victims/glow/libs/MC01/off/MC01_runner
WEIGHTS=victims/glow/libs/MC01/off/MC01.weights.bin
INPUT=data/inputs/lenet/mnist64/test/idx000000/input_nchw_f32.bin

setarch "$(uname -m)" -R env \
  GLOW_WRITEBACK_PROTECT=0 \
  GLOW_INPUT_ZERO_DITHER=0 \
  "$PIN" -t "$TOOL" \
    -stack-depth 0 \
    -taint-seed-mode file \
    -taint-file "$INPUT" \
    -o out/taint_block16_bits.json \
    -m out/taint_block16.ip.txt \
    -- "$RUNNER" "$WEIGHTS" "$INPUT" out/infer_result.txt
```

## Run IC (input-tensor seed)

```bash
bash scripts/trace/collect_rho_trans.sh ic IC01
# uses pintrace TOOL + image-classifier + PNG + -taint-seed-mode input-tensor
```

Keep the `pin/` directory hierarchy intact (`pin` next to `intel64/`).

## Rebuild the tool (optional)

```bash
cd pintrace/tool
make
```

Requires the shipped `pin/source/tools/Config` and `pin/source/include`.
Prebuilt `obj-intel64/taintblock16trace.so` is enough for normal use.
