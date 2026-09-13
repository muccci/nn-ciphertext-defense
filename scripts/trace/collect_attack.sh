#!/usr/bin/env bash
# Collect site-bit traces for attack recon / label.
#
#   bash scripts/trace/collect_attack.sh --case MC01 --mode off --site fc_110
#   bash scripts/trace/collect_attack.sh --case MC04 --mode on --site conv_960
#   bash scripts/trace/collect_attack.sh --case MC10 --mode off --site layer1_0_conv2
#   bash scripts/trace/collect_attack.sh --case MC15 --mode off --site transition1
# Full train/test loop: ./scripts/run_label_attack.sh
#
# Same pintool / ON env as collect_rho_trans.sh, plus -site-ip / -site-bits-only /
# -site-bits-txt. Always writes
#   artifact/trace_out/attack/<case>/<mode>/<site>/{train,test}/idx*/bits.txt
# Input from data/inputs/<family>/<variant>/<split>/idx*/ (collect_inputs.py).
# {train,test}/idx* is the sample's dataset position (not "one vs many").
#
# MC10: prefix 14108672 (16B-block pairs), then fold_odd → 7054336.
# MC15 --site is transition1|transition2|transition3; the three strings are
# assembled inside this script.
set -euo pipefail

TRACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd -- "$TRACE_DIR/.." && pwd)"
ART_ROOT="$(cd -- "$SCRIPTS_DIR/.." && pwd)"
TRACE_OUT="${TRACE_OUT:-$ART_ROOT/trace_out/attack}"
MANIFEST="${MANIFEST:-$ART_ROOT/victims/models/MANIFEST.txt}"
MODELS="${MODELS_DIR:-$ART_ROOT/victims/models}"
GLOW_LIBS="${BUNDLES_ROOT:-${GLOW_LIBS:-$ART_ROOT/victims/glow/libs}}"
GLOW_COMMON="${COMMON_DIR:-$ART_ROOT/victims/glow/common}"
PIN="${PIN:-$ART_ROOT/pintrace/pin/pin}"
TOOL="${TOOL:-$ART_ROOT/pintrace/tool/obj-intel64/taintblock16trace.so}"
FORCE="${FORCE:-0}"

FC_SYM="libjit_fc_f"
CONV_SYM="libjit_convDKKC8_convolve_channel"
OFF_FC=0x110
OFF_CONV960=0x960
MC15_WB0=0x97c
MC15_WB1=0x9b2
MC10_SLICE=14108672
MC10_FOLDED=7054336
MC15_SEG=4718592

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

die() { echo "error: $*" >&2; exit 1; }

bits_ready() {
  local out_dir="$1"
  [[ -s "$out_dir/bits.txt" || -s "$out_dir/bits.bin.gz" ]]
}

bits_count() {
  local dest="$1"
  python3 - "$dest" <<'PY'
import gzip
import sys
from pathlib import Path

dest = Path(sys.argv[1])
path = None
for name in ("bits.txt", "bits.bin.gz", "bits.bin"):
    cand = dest / name
    if cand.is_file() and cand.stat().st_size:
        path = cand
        break
if path is None:
    print(0)
    raise SystemExit(0)
raw = path.read_bytes()
if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
    raw = gzip.decompress(raw)
if raw and set(raw[: min(64, len(raw))]) <= {48, 49, 10, 13}:
    print(sum(1 for b in raw if b in (48, 49)))
else:
    print(len(raw) * 8)
PY
}

already_done() {
  local out_dir="$1"
  local want="${2:-0}"
  [[ "$FORCE" == "1" ]] && return 1
  bits_ready "$out_dir" || return 1
  [[ -f "$out_dir/run.log" ]] || return 1
  grep -q '^\[+\] done' "$out_dir/run.log" 2>/dev/null || return 1
  if [[ "$want" -gt 0 ]]; then
    [[ "$(bits_count "$out_dir")" -eq "$want" ]] || return 1
  fi
}

# Each 16B-block pair → the odd (second) bit. 00→0, 01→1, 10→0, 11→1.
fold_odd_bits() {
  local src="$1" dest="$2" take="$3"
  python3 - "$src" "$dest" "$take" <<'PY'
import gzip
import sys
from pathlib import Path

src, dest = Path(sys.argv[1]), Path(sys.argv[2])
take = int(sys.argv[3])
raw = src.read_bytes()
if src.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
    raw = gzip.decompress(raw)
if raw and set(raw[: min(64, len(raw))]) <= {48, 49, 10, 13}:
    bits = bytes(b for b in raw if b in (48, 49))
else:
    bits = bytes(((byte >> (7 - i)) & 1) + 48 for byte in raw for i in range(8))
if len(bits) < take:
    raise SystemExit(f"fold_odd need {take}, got {len(bits)}")
pair = bits[:take]
if len(pair) % 2:
    raise SystemExit(f"fold_odd odd length {len(pair)}")
dest.write_bytes(bytes(pair[i + 1] for i in range(0, len(pair), 2)))
PY
}

pack_ascii_bits() {
  local dest="$1"
  [[ -s "$dest/bits.txt" ]] || return 0
  python3 - "$dest/bits.txt" <<'PY'
import gzip
import sys
from pathlib import Path

src = Path(sys.argv[1])
raw = bytes(b for b in src.read_bytes() if b in (48, 49))
nbytes, rem = divmod(len(raw), 8)
packed = bytearray(nbytes + (1 if rem else 0))
acc = 0
nbit = 0
oi = 0
for b in raw:
    acc = (acc << 1) | (1 if b == 49 else 0)
    nbit += 1
    if nbit == 8:
        packed[oi] = acc
        oi += 1
        acc = 0
        nbit = 0
if nbit:
    packed[oi] = acc << (8 - nbit)
out = src.with_name("bits.bin.gz")
out.write_bytes(gzip.compress(bytes(packed), 1))
src.unlink()
PY
}

write_sample_sidecar() {
  local dest="$1"
  mkdir -p "$dest"
  ln -sfn "$INPUT_BIN" "$dest/input_nchw_f32.bin"
  python3 - "$dest" "$SPLIT" "$SAMPLE" "$MANIFEST_INPUT" "$(dirname "$INPUT_BIN")/meta.json" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
meta = {
    "split": sys.argv[2],
    "sample": sys.argv[3],
    "manifest_input": sys.argv[4],
}
src = Path(sys.argv[5])
if src.is_file():
    extra = json.loads(src.read_text(encoding="utf-8"))
    for key in (
        "label",
        "label_index",
        "gt_label",
        "gt_positive_indices",
        "patient_id",
        "filename",
        "source_index",
    ):
        if key in extra:
            meta[key] = extra[key]
(dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY
}

dither_thresh_for_case() {
  case "$1" in
    MC01) echo 1.0 ;;
    MC06|MC11|MC12|MC13|MC14) echo 4.0 ;;
    *) echo 1.0 ;;
  esac
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

nm_base_hex() {
  # Glow JIT names are often C++-mangled. Match exact $3, then substring.
  # Do not pipe nm into awk: with pipefail, awk exiting early makes nm
  # die SIGPIPE and this function false-fails even when the symbol exists.
  local runner="$1" needle="$2"
  local tmp hit
  tmp="$(mktemp)"
  if ! nm -n "$runner" >"$tmp"; then
    rm -f "$tmp"
    die "nm failed for $needle in $runner"
  fi
  hit="$(awk -v needle="$needle" '
    $3 == needle { print $1; exit }
    index($3, needle) { print $1; exit }
  ' "$tmp")"
  rm -f "$tmp"
  [[ -n "$hit" ]] || die "nm failed for $needle in $runner"
  printf '%s' "$hit"
}

site_ip_from() {
  local base_hex="$1" offset="$2"
  printf '0x%x' $((16#$base_hex + offset))
}

pin_site() {
  local out="$1" site_ip="$2"
  mkdir -p "$out"
  if [[ "$FORCE" != "1" ]] && bits_ready "$out"; then
    echo "[reuse] $out" | tee -a "$COLLECT_LOG"
    return 0
  fi
  local -a env_args=("$SETARCH_BIN" "$(uname -m)" -R env)
  if [[ "$MODE" == "off" ]]; then
    env_args+=(GLOW_WRITEBACK_PROTECT=0 GLOW_INPUT_ZERO_DITHER=0)
  else
    local -a on_env
    mapfile -t on_env < <(glow_on_env "$CASE")
    env_args+=("${on_env[@]}")
  fi
  "${env_args[@]}" \
    "$PIN" -t "$TOOL" \
      -site-ip "$site_ip" \
      -site-bits-only 1 \
      -site-bits-txt "$out/bits.txt" \
      -stack-depth 0 \
      -no-paddr 1 \
      -taint-seed-mode file \
      -taint-file "$INPUT_BIN" \
      -o "$out/site_bits.json" \
      -m "$out/site_ip.txt" \
      -- "$RUNNER" "$WEIGHTS" "$INPUT_BIN" "$out/infer_result.txt" \
      >"$out/pin.stdout.txt" 2>"$out/pin.stderr.txt"
  [[ -f "$out/bits.txt" ]] || die "pin wrote no bits: $out"
}

bits_len() {
  local path="$1"
  [[ -f "$path" ]] || { echo 0; return; }
  wc -c <"$path" | tr -d ' '
}

write_mc15_all_sites() {
  local a="$1" b="$2" mode_dir="$3" split="$4" sample="$5"
  python3 - "$a" "$b" "$mode_dir" "$MC15_SEG" "$split" "$sample" <<'PY'
from pathlib import Path
import sys

a, b, root, seg, split, sample = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
    int(sys.argv[4]), sys.argv[5], sys.argv[6],
)
x, y = a.read_bytes(), b.read_bytes()
n = min(len(x), len(y))
n -= n % 2
out = bytearray(n * 2)
i = oi = 0
while i < n:
    out[oi] = x[i]
    out[oi + 1] = x[i + 1]
    out[oi + 2] = y[i]
    out[oi + 3] = y[i + 1]
    i += 2
    oi += 4
if len(out) != 3 * seg:
    raise SystemExit(f"MC15 len={len(out)} want {3 * seg}")
for idx, name in enumerate(("transition1", "transition2", "transition3")):
    dest = root / name / split / sample
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bits.txt").write_bytes(bytes(out[idx * seg : (idx + 1) * seg]))
PY
}

CASE=""
MODE=""
SITE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    --case) CASE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --site) SITE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -n "$CASE" && -n "$MODE" && -n "$SITE" ]] || die "need --case MCxx --mode off|on --site <name>"
[[ "$MODE" == "off" || "$MODE" == "on" ]] || die "mode must be off|on"
case "$CASE/$SITE" in
  MC01/fc_110|MC04/conv_960|MC10/layer1_0_conv2) ;;
  MC15/transition1|MC15/transition2|MC15/transition3) ;;
  *) die "unsupported $CASE --site $SITE" ;;
esac

[[ -x "$PIN" ]] || die "missing pin: $PIN"
[[ -f "$TOOL" ]] || die "missing tool: $TOOL"
[[ -f "$MANIFEST" ]] || die "missing MANIFEST: $MANIFEST"
SETARCH_BIN="${SETARCH_BIN:-$(command -v setarch || true)}"
[[ -n "$SETARCH_BIN" ]] || die "setarch not found in PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

family_variant="$(awk -v c="$CASE" '$1==c {print $4; exit}' "$MANIFEST")"
[[ -n "$family_variant" ]] || die "MANIFEST missing $CASE"
family="${family_variant%%/*}"
rest="${family_variant#*/}"
variant="${rest%%/*}"
MANIFEST_INPUT="$(awk -v c="$CASE" '$1==c {print $5; exit}' "$MANIFEST")"
SPLIT="${SPLIT:-test}"
SAMPLE="${SAMPLE:-$MANIFEST_INPUT}"
if [[ "$SAMPLE" =~ ^idx[0-9]+$ ]]; then
  :
elif [[ "$SAMPLE" =~ ^[0-9]+$ ]]; then
  SAMPLE="$(printf 'idx%06d' "$((10#$SAMPLE))")"
else
  SAMPLE="idx000000"
fi
INPUT_BIN="$(python3 "$SCRIPTS_DIR/collect_inputs.py" resolve \
  --family "$family" --variant "$variant" --split "$SPLIT" --sample "$SAMPLE")"
RUNNER="$GLOW_LIBS/$CASE/$MODE/${CASE}_runner"
WEIGHTS="$GLOW_LIBS/$CASE/$MODE/${CASE}.weights.bin"
[[ -x "$RUNNER" ]] || die "missing runner $RUNNER"
[[ -f "$WEIGHTS" ]] || die "missing weights $WEIGHTS"
[[ -f "$INPUT_BIN" ]] || die "missing input $INPUT_BIN"

MODE_DIR="$TRACE_OUT/$CASE/$MODE"
OUT="$MODE_DIR/$SITE/$SPLIT/$SAMPLE"
if [[ "$CASE" == "MC15" ]]; then
  if already_done "$MODE_DIR/transition1/$SPLIT/$SAMPLE" "$MC15_SEG" \
     && already_done "$MODE_DIR/transition2/$SPLIT/$SAMPLE" "$MC15_SEG" \
     && already_done "$MODE_DIR/transition3/$SPLIT/$SAMPLE" "$MC15_SEG"; then
    echo "[skip] $CASE/$MODE/*/$SPLIT/$SAMPLE"
    exit 0
  fi
elif [[ "$CASE" == "MC01" ]] && already_done "$OUT" 335400; then
  echo "[skip] $CASE/$MODE/$SITE/$SPLIT/$SAMPLE"
  exit 0
elif [[ "$CASE" == "MC04" ]] && already_done "$OUT" 510464; then
  echo "[skip] $CASE/$MODE/$SITE/$SPLIT/$SAMPLE"
  exit 0
elif [[ "$CASE" == "MC10" ]] && already_done "$OUT" "$MC10_FOLDED"; then
  echo "[skip] $CASE/$MODE/$SITE/$SPLIT/$SAMPLE"
  exit 0
fi
mkdir -p "$OUT"
COLLECT_LOG="$OUT/run.log"

{
  echo "case=$CASE mode=$MODE site=$SITE"
  echo "split=$SPLIT sample=$SAMPLE manifest_input=$MANIFEST_INPUT"
  echo "runner=$RUNNER"
  echo "input=$INPUT_BIN"
  echo "pin=$PIN"
  echo "tool=$TOOL"
  echo "start=$(date -Is)"
} >"$COLLECT_LOG"

expect=0
case "$CASE" in
  MC01)
    base="$(nm_base_hex "$RUNNER" "$FC_SYM")"
    ip="$(site_ip_from "$base" "$OFF_FC")"
    echo "symbol=$FC_SYM offset=$OFF_FC nm_base=0x$base site_ip=$ip" >>"$OUT/run.log"
    pin_site "$OUT" "$ip"
    expect=335400
    ;;
  MC04)
    base="$(nm_base_hex "$RUNNER" "$CONV_SYM")"
    ip="$(site_ip_from "$base" "$OFF_CONV960")"
    echo "symbol=$CONV_SYM offset=$OFF_CONV960 nm_base=0x$base site_ip=$ip" >>"$OUT/run.log"
    pin_site "$OUT" "$ip"
    expect=510464
    ;;
  MC10)
    base="$(nm_base_hex "$RUNNER" "$CONV_SYM")"
    ip="$(site_ip_from "$base" "$OFF_CONV960")"
    echo "symbol=$CONV_SYM offset=$OFF_CONV960 nm_base=0x$base site_ip=$ip" >>"$OUT/run.log"
    have="$(bits_count "$OUT")"
    if [[ "$have" -eq "$MC10_SLICE" ]]; then
      echo "fold_odd reuse prefix=$have -> $MC10_FOLDED" >>"$OUT/run.log"
      src="$OUT/bits.txt"
      [[ -s "$src" ]] || src="$OUT/bits.bin.gz"
      fold_odd_bits "$src" "$OUT/bits.txt" "$MC10_SLICE"
      rm -f "$OUT/bits.bin.gz"
    else
      raw="$OUT/_conv960"
      pin_site "$raw" "$ip"
      raw_n="$(bits_len "$raw/bits.txt")"
      echo "raw_conv960_len=$raw_n slice=$MC10_SLICE fold_odd=$MC10_FOLDED" >>"$OUT/run.log"
      [[ "$raw_n" -ge "$MC10_SLICE" ]] || die "MC10 raw len $raw_n < $MC10_SLICE"
      head -c "$MC10_SLICE" "$raw/bits.txt" >"$raw/prefix.txt"
      fold_odd_bits "$raw/prefix.txt" "$OUT/bits.txt" "$MC10_SLICE"
      cp -f "$raw/site_ip.txt" "$OUT/site_ip.txt" 2>/dev/null || true
      rm -rf "$raw"
    fi
    expect="$MC10_FOLDED"
    ;;
  MC15)
    base="$(nm_base_hex "$RUNNER" "$CONV_SYM")"
    ip0="$(site_ip_from "$base" "$MC15_WB0")"
    ip1="$(site_ip_from "$base" "$MC15_WB1")"
    echo "symbol=$CONV_SYM nm_base=0x$base" >>"$COLLECT_LOG"
    work="$MODE_DIR/.wb/$SPLIT/$SAMPLE"
    pin_site "$work/0" "$ip0"
    pin_site "$work/1" "$ip1"
    write_mc15_all_sites "$work/0/bits.txt" "$work/1/bits.txt" "$MODE_DIR" "$SPLIT" "$SAMPLE"
    rm -rf "$work"
    expect="$MC15_SEG"
    ;;
esac

finish_one() {
  local dest="$1"
  local got
  got="$(bits_len "$dest/bits.txt")"
  write_sample_sidecar "$dest"
  echo "bits_len=$got expect=$expect" >>"$dest/run.log"
  echo "$dest/bits.txt len=$got expect=$expect"
  if [[ "$expect" -gt 0 && "$got" -ne "$expect" ]]; then
    echo "warn: length mismatch" >>"$dest/run.log"
    return 3
  fi
  pack_ascii_bits "$dest" || return 4
  echo "[+] done $(date -Is)" >>"$dest/run.log"
}

if [[ "$CASE" == "MC15" ]]; then
  ec=0
  for name in transition1 transition2 transition3; do
    dest="$MODE_DIR/$name/$SPLIT/$SAMPLE"
    mkdir -p "$dest"
    if [[ "$dest" != "$OUT" && -f "$COLLECT_LOG" ]]; then
      cp -f "$COLLECT_LOG" "$dest/run.log"
    fi
    finish_one "$dest" || ec=$?
  done
  exit "$ec"
fi
finish_one "$OUT"
