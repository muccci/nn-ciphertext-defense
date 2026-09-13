#!/usr/bin/env bash
# Collect Pin taintblock16 traces for ρ_trans (Glow MC/IC, TVM TV/TA).
#
#   bash scripts/trace/collect_rho_trans.sh mc
#   bash scripts/trace/collect_rho_trans.sh tv --case TV01
#   bash scripts/trace/collect_rho_trans.sh glow            # mc + ic
#   bash scripts/trace/collect_rho_trans.sh tvm             # tv + ta
#   bash scripts/trace/collect_rho_trans.sh all
#   bash scripts/trace/collect_rho_trans.sh ic IC01
#   bash scripts/trace/collect_rho_trans.sh mc --case MC01 --modes off
#   GLOW_RELU_PATCH_FIXED_VALUE=116 bash scripts/trace/collect_rho_trans.sh mc --modes on
#
# Input from scripts/collect_inputs.py (raw data/ → collect files).
# MC/TV/TA: input_nchw_f32.bin + Pin -taint-file.
# IC: input_n0.png + image_mode.txt + Pin input-tensor (same PNG encode as before).
# Writes artifact/trace_out/rho_trans/{mc,ic,tv,ta}/<id>/{off,on}/{train,test}/idx*/.
# {train,test}/idx* is the sample's dataset position (same convention as attack).
# Attack site-bit collect is scripts/trace/collect_attack.sh.
set -euo pipefail

TRACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd -- "$TRACE_DIR/.." && pwd)"
ART_ROOT="$(cd -- "$SCRIPTS_DIR/.." && pwd)"

MANIFEST="${MANIFEST:-$ART_ROOT/victims/models/MANIFEST.txt}"
MODELS="${MODELS_DIR:-$ART_ROOT/victims/models}"
GLOW_DIR="${GLOW_DIR:-$ART_ROOT/victims/glow}"
GLOW_LIBS="${BUNDLES_ROOT:-${GLOW_LIBS:-$GLOW_DIR/libs}}"
GLOW_COMMON="${COMMON_DIR:-$GLOW_DIR/common}"
TVM_DIR="${TVM_DIR:-$ART_ROOT/victims/tvm}"
TVM_LIBS="${TVM_LIBS:-$TVM_DIR/libs}"
TVM_COMMON="${TVM_COMMON:-$TVM_DIR/common}"
TVM_BUILD="${TVM_BUILD:-$ART_ROOT/writeback-protect/tvm/build-llvm14-glow}"
VM_RUNNER="${VM_RUNNER:-$TVM_DIR/bin/generic_tvm_native_vm_runner}"
PIN="${PIN:-$ART_ROOT/pintrace/pin/pin}"
TOOL="${TOOL:-$ART_ROOT/pintrace/tool/obj-intel64/taintblock16trace.so}"
IC_BIN="${IC_BIN:-$ART_ROOT/writeback-protect/glow/build_ic/bin/image-classifier}"
PIN_TRACES="${PIN_TRACES:-$ART_ROOT/trace_out/rho_trans}"
MNIST_01_BIN="${MNIST_01_BIN:-}"
PAR="${PAR:-4}"
FORCE="${FORCE:-0}"
MODES="${MODES:-off,on}"

die() { echo "error: $*" >&2; exit 1; }

to_idx() {
  local raw="${1:-}"
  if [[ "$raw" =~ ^idx[0-9]+$ ]]; then
    printf '%s' "$raw"
  elif [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf 'idx%06d' "$((10#$raw))"
  else
    printf 'idx000000'
  fi
}

write_meta() {
  local dest="$1" sample="$2"
  printf '{\n  "split": "%s",\n  "sample": "%s"\n}\n' \
    "${SPLIT:-test}" "$sample" >"$dest/meta.json"
}

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

trace_root_for() {
  case "$1" in
    mc|ic|tv|ta) printf '%s' "$PIN_TRACES/$1" ;;
    *) die "unknown backend: $1" ;;
  esac
}

dither_thresh_for_case() {
  case "$1" in
    MC01|IC01) echo 1.0 ;;
    MC06|MC11|MC12|MC13|MC14|IC06|IC11|IC12|IC13|IC14) echo 4.0 ;;
    *) echo 1.0 ;;
  esac
}

already_done() {
  local out_dir="$1"
  [[ "$FORCE" == "1" ]] && return 1
  [[ -f "$out_dir/taint_block16_bits.json" && -f "$out_dir/run.log" ]] || return 1
  grep -q '^\[+\] done' "$out_dir/run.log" 2>/dev/null
}

glow_on_env() {
  local case_id="$1"
  # shellcheck disable=SC1091
  source "$GLOW_COMMON/wb_protect_compile_env.sh"
  local thresh
  thresh="$(dither_thresh_for_case "$case_id")"
  printf '%s\n' \
    GLOW_WRITEBACK_PROTECT=1 \
    GLOW_INPUT_ZERO_DITHER=random \
    GLOW_INPUT_ZERO_DITHER_LAYOUT=NCHW \
    GLOW_INPUT_ZERO_DITHER_EPS_MIN=1e-5 \
    GLOW_INPUT_ZERO_DITHER_EPS_MAX=2e-5 \
    "GLOW_INPUT_ZERO_DITHER_THRESH=$thresh" \
    GLOW_INPUT_ZERO_DITHER_SEED=1234 \
    "GLOW_RELU_PATCH_BITS=${GLOW_RELU_PATCH_BITS}" \
    "GLOW_RELU_PATCH_FIXED_BITS=${GLOW_RELU_PATCH_FIXED_BITS}" \
    "GLOW_RELU_PATCH_FIXED_VALUE=${GLOW_RELU_PATCH_FIXED_VALUE}" \
    "GLOW_RELU_PATCH_INC=${GLOW_RELU_PATCH_INC}" \
    "GLOW_RELU_PATCH_POSITIVE=${GLOW_RELU_PATCH_POSITIVE}"
}

tvm_on_env() {
  # shellcheck disable=SC1091
  source "$TVM_COMMON/wb_protect_compile_env.sh"
  printf '%s\n' \
    TVM_WRITEBACK_PROTECT=1 \
    GLOW_WRITEBACK_PROTECT=1 \
    TVM_RELU_LOW12_PATCH=1 \
    "TVM_RELU_PATCH_BITS=${TVM_RELU_PATCH_BITS}" \
    "TVM_RELU_PATCH_FIXED_BITS=${TVM_RELU_PATCH_FIXED_BITS}" \
    "TVM_RELU_PATCH_FIXED_VALUE=${TVM_RELU_PATCH_FIXED_VALUE}" \
    "TVM_RELU_PATCH_INC=${TVM_RELU_PATCH_INC}" \
    "TVM_RELU_PATCH_POSITIVE=${TVM_RELU_PATCH_POSITIVE}" \
    "TVM_INPUT_ZERO_DITHER=${TVM_INPUT_ZERO_DITHER}" \
    "TVM_INPUT_ZERO_DITHER_LAYOUT=${TVM_INPUT_ZERO_DITHER_LAYOUT}" \
    "TVM_INPUT_ZERO_DITHER_THRESH=${TVM_INPUT_ZERO_DITHER_THRESH}" \
    "TVM_INPUT_ZERO_DITHER_EPS_MIN=${TVM_INPUT_ZERO_DITHER_EPS_MIN}" \
    "TVM_INPUT_ZERO_DITHER_EPS_MAX=${TVM_INPUT_ZERO_DITHER_EPS_MAX}" \
    "TVM_INPUT_ZERO_DITHER_SILENT=${TVM_INPUT_ZERO_DITHER_SILENT:-0}" \
    "GLOW_INPUT_ZERO_DITHER=${GLOW_INPUT_ZERO_DITHER}" \
    "GLOW_INPUT_ZERO_DITHER_LAYOUT=${GLOW_INPUT_ZERO_DITHER_LAYOUT}" \
    "GLOW_INPUT_ZERO_DITHER_THRESH=${GLOW_INPUT_ZERO_DITHER_THRESH}" \
    "GLOW_INPUT_ZERO_DITHER_EPS_MIN=${GLOW_INPUT_ZERO_DITHER_EPS_MIN}" \
    "GLOW_INPUT_ZERO_DITHER_EPS_MAX=${GLOW_INPUT_ZERO_DITHER_EPS_MAX}"
}

resolve_input() {
  local family="$1" variant="$2" split="$3" sample="$4" kind="${5:-bin}"
  local -a extra=()
  if [[ -n "$MNIST_01_BIN" && -f "$MNIST_01_BIN" ]]; then
    extra+=(--zero-one-bin "$MNIST_01_BIN")
  fi
  python3 "$SCRIPTS_DIR/collect_inputs.py" resolve \
    --family "$family" --variant "$variant" \
    --split "$split" --sample "$sample" --kind "$kind" "${extra[@]}"
}

run_one_mc() {
  local mc="$1" family="$2" variant="$3" sample="$4" mode="$5"
  local runner="$GLOW_LIBS/$mc/$mode/${mc}_runner"
  local weights="$GLOW_LIBS/$mc/$mode/${mc}.weights.bin"
  local input
  input="$(resolve_input "$family" "$variant" "${SPLIT:-test}" "$sample")"
  local out_dir="$OUT_ROOT/$mc/$mode/${SPLIT:-test}/$sample"
  mkdir -p "$out_dir"
  if already_done "$out_dir"; then echo "[skip] $mc/$mode/${SPLIT:-test}/$sample"; return 0; fi
  [[ -x "$runner" ]] || die "missing runner $runner"
  [[ -f "$weights" ]] || die "missing weights $weights"
  [[ -f "$input" ]] || die "missing input $input"

  {
    echo "backend=mc"; echo "mc=$mc"; echo "mode=$mode"
    echo "split=${SPLIT:-test}"; echo "sample=$sample"
    echo "runner=$runner"; echo "input=$input"; echo "start=$(date -Is)"
  } >"$out_dir/run.log"

  local -a env_args=("$SETARCH_BIN" "$(uname -m)" -R env)
  if [[ "$mode" == "off" ]]; then
    env_args+=(GLOW_WRITEBACK_PROTECT=0 GLOW_INPUT_ZERO_DITHER=0)
  else
    mapfile -t _on < <(glow_on_env "$mc")
    env_args+=("${_on[@]}")
  fi

  echo "==> $mc $mode ($family/$variant)" | tee -a "$out_dir/run.log"
  "${env_args[@]}" \
    "$PIN" -t "$TOOL" -stack-depth 0 \
      -taint-seed-mode file -taint-file "$input" \
      -o "$out_dir/taint_block16_bits.json" \
      -m "$out_dir/taint_block16.ip.txt" \
      -- "$runner" "$weights" "$input" "$out_dir/infer_result.txt" \
      >"$out_dir/pin.stdout.txt" 2>"$out_dir/pin.stderr.txt"
  echo "[+] done $(date -Is) size=$(stat -c%s "$out_dir/taint_block16_bits.json" 2>/dev/null || echo 0)" \
    | tee -a "$out_dir/run.log"
  ln -sfn "$input" "$out_dir/input_nchw_f32.bin"
  write_meta "$out_dir" "$sample"
}

run_one_ic() {
  local ic="$1" onnx_rel="$2" mode="$3" sample="$4"
  local out_dir="$OUT_ROOT/$ic/$mode/${SPLIT:-test}/$sample"
  local onnx="$MODELS/$onnx_rel"
  mkdir -p "$out_dir"
  if already_done "$out_dir"; then echo "[skip] $ic/$mode/${SPLIT:-test}/$sample"; return 0; fi
  local family variant img0 image_mode
  family="${onnx_rel%%/*}"
  variant="$(printf '%s' "$onnx_rel" | cut -d/ -f2)"
  img0="$(resolve_input "$family" "$variant" "${SPLIT:-test}" "$sample" png)"
  image_mode="$(tr -d '\n' <"$(dirname "$img0")/image_mode.txt")"
  [[ -f "$img0" ]] || die "missing png: $img0"

  {
    echo "backend=ic"; echo "ic=$ic"; echo "mode=$mode"
    echo "split=${SPLIT:-test}"; echo "sample=$sample"
    echo "onnx=$onnx"; echo "png=$img0"; echo "start=$(date -Is)"
  } >"$out_dir/run.log"

  local -a env_args=("$SETARCH_BIN" "$(uname -m)" -R env LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
  if [[ "$mode" == "off" ]]; then
    env_args+=(GLOW_WRITEBACK_PROTECT=0 GLOW_INPUT_ZERO_DITHER=0)
  else
    mapfile -t _on < <(glow_on_env "$ic")
    env_args+=("${_on[@]}")
  fi

  local -a channel_args=()
  local ch
  ch="$(python3 -c "import onnx; m=onnx.load('$onnx'); print(m.graph.input[0].type.tensor_type.shape.dim[1].dim_value)")"
  [[ "$ch" == "3" ]] && channel_args+=(-image-channel-order=RGB)

  echo "==> $ic $mode png=$(basename "$img0")" | tee -a "$out_dir/run.log"
  if ! "${env_args[@]}" \
    "$PIN" -t "$TOOL" -stack-depth 0 -taint-seed-mode input-tensor \
      -o "$out_dir/taint_block16_bits.json" \
      -m "$out_dir/taint_block16.ip.txt" \
      -- "$IC_BIN" "$img0" \
        -model="$onnx" -model-input-name=data -output-name=output \
        -image-mode="$image_mode" "${channel_args[@]}" \
        -image-layout=NCHW -input-layout=NCHW -backend=Interpreter \
        -minibatch=1 -topk=1 \
    >"$out_dir/pin.stdout.txt" 2>"$out_dir/pin.stderr.txt"
  then
    die "pin failed for $ic/$mode (see $out_dir/pin.stderr.txt)"
  fi
  echo "[+] done $(date -Is) size=$(stat -c%s "$out_dir/taint_block16_bits.json" 2>/dev/null || echo 0)" \
    | tee -a "$out_dir/run.log"
  write_meta "$out_dir" "$sample"
}

run_one_tv() {
  local tv="$1" family="$2" variant="$3" mode="$4" sample="$5"
  local lib="$TVM_LIBS/$tv/$mode/${tv}_tvm.so"
  local shape="$TVM_LIBS/$tv/$mode/input_shape.txt"
  local input
  input="$(resolve_input "$family" "$variant" "${SPLIT:-test}" "$sample")"
  local out_dir="$OUT_ROOT/$tv/$mode/${SPLIT:-test}/$sample"
  mkdir -p "$out_dir"
  if already_done "$out_dir"; then echo "[skip] $tv/$mode/${SPLIT:-test}/$sample"; return 0; fi
  [[ -x "$VM_RUNNER" ]] || die "missing VM runner $VM_RUNNER (victims/tvm/compile.sh --runtime tv)"
  [[ -f "$lib" ]] || die "missing $lib"
  [[ -f "$shape" ]] || die "missing $shape"
  [[ -f "$input" ]] || die "missing $input"

  {
    echo "backend=tv"; echo "tv=$tv"; echo "mode=$mode"
    echo "split=${SPLIT:-test}"; echo "sample=$sample"
    echo "library=$lib"; echo "input=$input"; echo "shape=$(cat "$shape")"
    echo "start=$(date -Is)"
  } >"$out_dir/run.log"

  local -a env_args=(
    "$SETARCH_BIN" "$(uname -m)" -R env
    "LD_LIBRARY_PATH=$TVM_BUILD:$TVM_BUILD/lib:$LD_LIBRARY_PATH"
  )
  if [[ "$mode" == "off" ]]; then
    env_args+=(TVM_WRITEBACK_PROTECT=0 GLOW_WRITEBACK_PROTECT=0 TVM_INPUT_ZERO_DITHER=0 GLOW_INPUT_ZERO_DITHER=0)
  else
    mapfile -t _on < <(tvm_on_env)
    env_args+=("${_on[@]}")
  fi

  echo "==> $tv $mode ($family/$variant)" | tee -a "$out_dir/run.log"
  "${env_args[@]}" \
    "$PIN" -t "$TOOL" -stack-depth 0 -no-paddr 1 \
      -taint-seed-mode file -taint-file "$input" \
      -o "$out_dir/taint_block16_bits.json" \
      -m "$out_dir/taint_block16.ip.txt" \
      -- "$VM_RUNNER" \
        --library "$lib" \
        --input-bin "$input" \
        --input-shape "$(tr -d '[:space:]' <"$shape")" \
        --output-bin "$out_dir/output.bin" \
        --summary-json "$out_dir/summary.json" \
      >"$out_dir/pin.stdout.txt" 2>"$out_dir/pin.stderr.txt"
  echo "[+] done $(date -Is) size=$(stat -c%s "$out_dir/taint_block16_bits.json" 2>/dev/null || echo 0)" \
    | tee -a "$out_dir/run.log"
  ln -sfn "$input" "$out_dir/input_nchw_f32.bin"
  write_meta "$out_dir" "$sample"
}

run_one_ta() {
  local ta="$1" family="$2" variant="$3" mode="$4" sample="$5"
  local lib="$TVM_LIBS/$ta/$mode/${ta}_tvm_aot_kernels.so"
  local constants="$TVM_LIBS/$ta/$mode/constants"
  local runner="$TVM_LIBS/$ta/${ta}_tvm_aot_runner"
  local input
  input="$(resolve_input "$family" "$variant" "${SPLIT:-test}" "$sample")"
  local out_dir="$OUT_ROOT/$ta/$mode/${SPLIT:-test}/$sample"
  mkdir -p "$out_dir"
  if already_done "$out_dir"; then echo "[skip] $ta/$mode/${SPLIT:-test}/$sample"; return 0; fi
  [[ -x "$runner" ]] || die "missing AOT runner $runner (victims/tvm/compile.sh --runtime ta)"
  [[ -f "$lib" ]] || die "missing $lib"
  [[ -d "$constants" ]] || die "missing $constants"
  [[ -f "$input" ]] || die "missing $input"

  {
    echo "backend=ta"; echo "ta=$ta"; echo "mode=$mode"
    echo "split=${SPLIT:-test}"; echo "sample=$sample"
    echo "library=$lib"; echo "input=$input"; echo "start=$(date -Is)"
  } >"$out_dir/run.log"

  local -a env_args=(
    "$SETARCH_BIN" "$(uname -m)" -R env
    "LD_LIBRARY_PATH=$TVM_BUILD:$TVM_BUILD/lib:$LD_LIBRARY_PATH"
  )
  if [[ "$mode" == "off" ]]; then
    env_args+=(TVM_WRITEBACK_PROTECT=0 GLOW_WRITEBACK_PROTECT=0 TVM_INPUT_ZERO_DITHER=0 GLOW_INPUT_ZERO_DITHER=0)
  else
    mapfile -t _on < <(tvm_on_env)
    env_args+=("${_on[@]}")
  fi

  echo "==> $ta $mode ($family/$variant)" | tee -a "$out_dir/run.log"
  "${env_args[@]}" \
    "$PIN" -t "$TOOL" -stack-depth 0 -no-paddr 1 \
      -taint-seed-mode file -taint-file "$input" \
      -o "$out_dir/taint_block16_bits.json" \
      -m "$out_dir/taint_block16.ip.txt" \
      -- "$runner" \
        --library "$lib" \
        --constants-dir "$constants" \
        --input-bin "$input" \
        --output-bin "$out_dir/output.bin" \
        --summary-json "$out_dir/summary.json" \
      >"$out_dir/pin.stdout.txt" 2>"$out_dir/pin.stderr.txt"
  echo "[+] done $(date -Is) size=$(stat -c%s "$out_dir/taint_block16_bits.json" 2>/dev/null || echo 0)" \
    | tee -a "$out_dir/run.log"
  ln -sfn "$input" "$out_dir/input_nchw_f32.bin"
  write_meta "$out_dir" "$sample"
}

id_for_backend() {
  local backend="$1" mc="$2"
  case "$backend" in
    mc) printf '%s' "$mc" ;;
    ic) printf 'IC%s' "${mc#MC}" ;;
    tv) printf 'TV%s' "${mc#MC}" ;;
    ta) printf 'TA%s' "${mc#MC}" ;;
  esac
}

collect_one_backend() {
  local backend="$1"
  shift
  local -a filters=("$@")
  OUT_ROOT="$(trace_root_for "$backend")"
  if [[ -n "${TRACE_ROOT:-}" && ${#STAGES[@]} -eq 1 ]]; then
    OUT_ROOT="$TRACE_ROOT"
  fi

  case "$backend" in
    ic)
      [[ -x "$IC_BIN" ]] || die "missing image-classifier: $IC_BIN"
      ;;
    tv)
      [[ -x "$VM_RUNNER" ]] || die "missing $VM_RUNNER"
      ;;
    ta)
      [[ -d "$TVM_LIBS" ]] || die "missing $TVM_LIBS"
      ;;
  esac

  local jobs=() mc _tg _sha onnx_rel _sample local_id family variant match f
  while read -r mc _tg _sha onnx_rel _sample || [[ -n "${mc:-}" ]]; do
    [[ -z "${mc:-}" || "$mc" == \#* ]] && continue
    [[ "$mc" == MC* ]] || continue
    local_id="$(id_for_backend "$backend" "$mc")"
    if [[ ${#filters[@]} -gt 0 ]]; then
      match=0
      for f in "${filters[@]}"; do
        # Exact id / MC number only. Do not substring-match onnx paths
        # ("11" used to hit densenet121 → TV15 and abort the TA stage).
        if [[ "$local_id" == "$f" || "$mc" == "$f" || "$f" == "${mc#MC}" ]]; then
          match=1
          break
        fi
        if [[ "$f" =~ ^(MC|IC|TV|TA)?([0-9]+)$ ]]; then
          fn="$(printf '%02d' "$((10#${BASH_REMATCH[2]}))")"
          if [[ "${mc#MC}" == "$fn" ]]; then
            match=1
            break
          fi
        elif [[ "$f" == */* && "$onnx_rel" == *"$f"* ]]; then
          match=1
          break
        fi
      done
      [[ "$match" == 1 ]] || continue
    fi
    family="${onnx_rel%%/*}"
    variant="$(printf '%s' "$onnx_rel" | cut -d/ -f2)"
    sample="$(to_idx "${SAMPLE:-$_sample}")"
    local raw_mode
    local -a mode_list=()
    IFS=',' read -ra _mode_items <<< "$MODES"
    for raw_mode in "${_mode_items[@]}"; do
      raw_mode="${raw_mode// /}"
      [[ "$raw_mode" == off || "$raw_mode" == on ]] || die "mode must be off|on (got $raw_mode)"
      mode_list+=("$raw_mode")
    done
    for mode in "${mode_list[@]}"; do
      case "$backend" in
        mc) jobs+=("mc|$local_id|$family|$variant|$sample|$mode") ;;
        ic) jobs+=("ic|$local_id|$onnx_rel|$mode|$sample") ;;
        tv) jobs+=("tv|$local_id|$family|$variant|$mode|$sample") ;;
        ta) jobs+=("ta|$local_id|$family|$variant|$mode|$sample") ;;
      esac
    done
  done <"$MANIFEST"

  [[ ${#jobs[@]} -gt 0 ]] || die "no MANIFEST rows matched for $backend: ${filters[*]:-<all>}"
  echo "backend=$backend jobs=${#jobs[@]} PAR=$PAR OUT_ROOT=$OUT_ROOT"

  export -f run_one_mc run_one_ic run_one_tv run_one_ta \
    resolve_input dither_thresh_for_case glow_on_env tvm_on_env already_done die \
    to_idx write_meta
  export ART_ROOT OUT_ROOT PIN TOOL MODELS FORCE SETARCH_BIN LD_LIBRARY_PATH
  export GLOW_LIBS GLOW_COMMON IC_BIN MNIST_01_BIN
  export TVM_LIBS TVM_COMMON TVM_BUILD VM_RUNNER SPLIT SCRIPTS_DIR

  printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
    IFS="|" read -r kind a b c d e <<<"$1"
    case "$kind" in
      mc) run_one_mc "$a" "$b" "$c" "$d" "$e" ;;
      ic) run_one_ic "$a" "$b" "$c" "$d" ;;
      tv) run_one_tv "$a" "$b" "$c" "$d" "$e" ;;
      ta) run_one_ta "$a" "$b" "$c" "$d" "$e" ;;
    esac
  ' _ {} || return 1

  echo "collect done. OUT_ROOT=$OUT_ROOT"
  echo "next: python3 scripts/compute_victim_rho_trans.py --backend $backend --trace-root $OUT_ROOT"
}

# ---- args ----
BACKEND="${BACKEND:-}"
FILTERS=()
BACKEND_FROM_ARG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --backend) BACKEND="$2"; BACKEND_FROM_ARG=1; shift 2 ;;
    --case) FILTERS+=("$2"); shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    mc|ic|tv|ta|glow|tvm|all)
      if [[ "$BACKEND_FROM_ARG" -eq 0 && ${#FILTERS[@]} -eq 0 ]]; then
        BACKEND="$1"
        BACKEND_FROM_ARG=1
      else
        FILTERS+=("$1")
      fi
      shift
      ;;
    *) FILTERS+=("$1"); shift ;;
  esac
done

[[ -n "$BACKEND" ]] || die "need backend: mc|ic|tv|ta|glow|tvm|all"
SPLIT="${SPLIT:-test}"
export SPLIT
[[ -x "$PIN" ]] || die "missing pin: $PIN"
[[ -f "$TOOL" ]] || die "missing tool: $TOOL"
[[ -f "$MANIFEST" ]] || die "missing MANIFEST: $MANIFEST"
SETARCH_BIN="${SETARCH_BIN:-$(command -v setarch || true)}"
[[ -n "$SETARCH_BIN" ]] || die "setarch not found in PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

expand_backends() {
  case "$1" in
    glow) printf '%s\n' mc ic ;;
    tvm) printf '%s\n' tv ta ;;
    all) printf '%s\n' mc ic tv ta ;;
    mc|ic|tv|ta) printf '%s\n' "$1" ;;
    *) die "backend must be mc|ic|tv|ta|glow|tvm|all (got: $1)" ;;
  esac
}

mapfile -t STAGES < <(expand_backends "$BACKEND")
ec=0
for stage in "${STAGES[@]}"; do
  if ! collect_one_backend "$stage" "${FILTERS[@]}"; then
    echo "error: collect stage=$stage failed" >&2
    ec=1
  fi
done
exit "$ec"
