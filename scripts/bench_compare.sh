#!/usr/bin/env bash
# =============================================================================
# bench_compare.sh — reproduce the README "llm.asm vs llama.cpp" CPU decode
# benchmark on ANY machine, and emit a ready-to-paste markdown table.
#
# Answers the standing request in issue #8 ("Request: llama-bench command and
# script for the Q8_0 comparison table") with a script instead of a table:
#   * llama.cpp side:  llama-bench -p 34 -n 64 -r 3 -ngl 0 (avg ± sd from its
#     own repetitions, markdown output parsed cell-exact)
#   * llm.asm side:    asm-llm --tokens <exact prompt ids> -n 64, one run per
#     repetition, decode tok/s parsed from its own "[asm-llm] decode :"
#     performance line; mean ± sd computed here with awk
#   * both sides use the SAME prompt token count and generated token count,
#     so the decode (tg) column is directly comparable token-for-token.
#
# Prefill is intentionally NOT benchmarked 1:1: llama.cpp prefills with
# batched GEMM while llm.asm prefills token-by-token (documented in both
# READMEs). The decodable claim — "tg tok/s" — is what this table certifies.
# =============================================================================
set -euo pipefail

# ---------- defaults ----------
PP=34                 # prompt tokens (matches the README table)
NG=64                 # generated tokens (matches the README table)
THREADS=""            # unset => derived from nproc
REPS=3
NGL=0                 # -ngl for llama-bench: 0 == pure CPU, as in the README
ASM_BIN=""
LLAMA_BENCH=""
ASM_MODEL=""
GGUF=""
OUTDIR=""
LIST_FIRST_TOKEN=1000 # deterministic prompt ids: 1000..(1000+PP-1)
DRY_RUN=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  -m <asmbin>          .asmbin model for asm-llm            (required, or --asm-model)
  -g <gguf>            GGUF model for llama-bench            (required, or --gguf)
  --asm-bin <path>     asm-llm binary      (default: auto-search build/bin, ., $PATH)
  --llama-bench <path> llama-bench binary  (default: auto-search build/bin dirs, $PATH)
  --asm-model <path>   same as -m
  --gguf <path>        same as -g
  -t "1,4,8,16,24"     comma-separated thread counts (default: 1,2,4,...40 capped at nproc)
  -r <reps>            repetitions per thread count (default: 3; llama-bench side is done
                       inside one invocation via its own -r)
  --no-warmup          skip the untimed warmup run on the asm side (default: warmup run first,
                       mirrors llama-bench's own internal warmup)
  -p <tokens>          prompt tokens   (default: 34)
  -n <tokens>          generated tokens (default: 64)
  --ngl <n>            llama-bench -ngl override (default: 0, CPU-only as in the README)
  --out <dir>          results directory (default: results/)
  --quick              1 rep per thread count
  --dry-run            print every command that would run, then exit
  -h | --help          this help
USAGE
  exit "${1:-0}"
}

die()  { printf 'bench_compare: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*" >&2; }

# ---------- args ----------
while [ $# -gt 0 ]; do
  case "$1" in
    -m|--asm-model) ASM_MODEL="$2"; shift 2;;
    -g|--gguf)      GGUF="$2"; shift 2;;
    --asm-bin)      ASM_BIN="$2"; shift 2;;
    --llama-bench)  LLAMA_BENCH="$2"; shift 2;;
    -t)             THREADS="$2"; shift 2;;
    -r)             REPS="$2"; shift 2;;
    -p)             PP="$2"; shift 2;;
    -n)             NG="$2"; shift 2;;
    --ngl)          NGL="$2"; shift 2;;
    --out)          OUTDIR="$2"; shift 2;;
    --quick)        REPS=1; shift;;
    --no-warmup)    NO_WARMUP=1; shift;;
    --dry-run)      DRY_RUN=1; shift;;
    -h|--help)      usage 0;;
    *) usage "unknown option: $1" 1 1>&2; exit 1;;
  esac
done

[ -n "$ASM_MODEL" ] || die "-m/--asm-model <path.to.asmbin> is required"
[ -n "$GGUF" ]      || die "-g/--gguf <path.to.gguf> is required"
[ -f "$ASM_MODEL" ] || die "asmbin model not found: $ASM_MODEL"
[ -f "$GGUF" ]      || die "gguf not found: $GGUF"
case "$PP $NG $REPS $NGL" in *[!0-9\ ]*) die "-p/-n/-r/--ngl must be integers";; esac
[ "$PP" -ge 1 ] && [ "$NG" -ge 1 ] && [ "$REPS" -ge 1 ] || die "token/reps values must be >= 1"

# ---------- preflight ----------
info "== preflight =="
if [ ! -r /proc/cpuinfo ]; then info "cpuinfo unreadable; SIMD not verified"; else
  FLAGS=$(grep -m1 '^flags' /proc/cpuinfo)
  for need in avx2 fma f16c; do
    grep -qw "$need" <<<"$FLAGS" || die "CPU lacks $need — both binaries require AVX2+FMA+F16C (see llm.asm README: Haswell/Excavator or newer)"
  done
  info "cpu flags ok: avx2 + fma + f16c"
fi

# ---------- binaries ----------
find_bin() { # $1=name $2=default search dirs...
  local name="$1"; local -a dirs=("${@:2}")
  if [ -n "$SEARCH_BIN" ]; then
    [ -x "$SEARCH_BIN" ] && { printf '%s' "$SEARCH_BIN"; return 0; }
    die "$name: not executable: $SEARCH_BIN"
  fi
  local d
  for d in "${dirs[@]}"; do
    # honor explicit dir-only values too (d may end in the binary name)
    if [ -x "$d" ]; then printf '%s' "$d"; return 0; fi
    if [ -x "$d/$name" ]; then printf '%s' "$d/$name"; return 0; fi
  done
  command -v "$name" || die "$name not found — build it (make inference / cmake --build ... --target llama-bench) or pass --asm-bin/--llama-bench"
}

# per-bin search dirs
SEARCH_BIN="$ASM_BIN"
ASM_EXEC=$(find_bin "asm-llm" "build/bin/asm-llm" "01-qwen3-inference/build/asm-llm" "build/asm-llm" "llm.asm/build/asm-llm" "../llm.asm/build/asm-llm")
SEARCH_BIN="$LLAMA_BENCH"
BENCH_EXEC=$(find_bin "llama-bench" "llama.cpp/build/bin/llama-bench" "build/bin/llama-bench" "../llama.cpp/build/bin/llama-bench")
[ -x "$ASM_EXEC" ]   || die "asm-llm not executable: $ASM_EXEC"
[ -x "$BENCH_EXEC" ] || die "llama-bench not executable: $BENCH_EXEC"
info "asm-llm   : $ASM_EXEC"
info "llama-bench: $BENCH_EXEC"

# ---------- threads ----------
if [ -z "$THREADS" ]; then
  HW=$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo)
  list=1
  for t in 2 4 8 16 24 32 40; do
    if [ "$t" -le "$HW" ]; then list="$list,$t"; fi
  done
  THREADS="$list"
fi
IFS=',' read -r -a TLIST <<<"$THREADS"
info "threads: ${TLIST[*]}"

# ---------- prompt token ids ----------
# exact-comparable prompt: deterministic pseudo-random token ids, no chat template
# (llama-bench also feeds raw ids). Diversity matters: repetitive ids distort timing.
TOK_IDS=$(awk -v n="$PP" 'BEGIN{
  for (i=0;i<n;i++) {
    id = 1000 + int((i*1103515245 + 73126469) % 148000)
    printf "%s%s", (i ? "," : ""), id
  }
}')
info "prompt: $PP deterministic pseudo-random raw token ids (no template) + $NG generated"

# ---------- output ----------
if [ "$DRY_RUN" -eq 1 ]; then
  info "== dry-run: commands that would execute =="
  for t in "${TLIST[@]}"; do
    info "\$ $BENCH_EXEC -m \"$GGUF\" -p $PP -n $NG -t $t -r $REPS -ngl $NGL -o md"
    for rep in $(seq "$REPS"); do
      info "\$ $ASM_EXEC -m \"$ASM_MODEL\" --tokens \"$TOK_IDS\" -n $NG -t $t -c $((PP+NG+64))"
    done
  done
  info "dry-run complete (nothing executed)"
  exit 0
fi

OUTDIR="${OUTDIR:-results}"
mkdir -p "$OUTDIR"
NOW=$(date +%Y%m%d-%H%M%S)
OUT_FILE="$OUTDIR/bench-compare-$NOW.md"
RAW_ASME="$OUTDIR/asm-llm-p34-n64-t$PP.$NOW.log"
: > "$RAW_ASME"

# ---------- machine info ----------
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs || true)
HW_TOTAL=$(nproc 2>/dev/null || echo '?')
KB_MEM=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
GB_MEM=$(( KB_MEM / 1024 / 1024 ))
KERNEL=$(uname -sr)

{
  echo "| | |"
  echo "|---|---|"
  echo "| CPU | \`$CPU_MODEL\` |"
  echo "| threads (hw) | $HW_TOTAL |"
  echo "| memory | ${GB_MEM} GiB |"
  echo "| kernel | $KERNEL |"
  echo "| models | asm: \`$ASM_MODEL\` — gguf: \`$GGUF\` |"
  echo "| test | $PP prompt tokens, $NG generated, -ngl $NGL, $REPS reps |"
  echo ""
  echo "| Threads | llama.cpp tg$NG tok/s (avg ± sd) | llm.asm tg$NG tok/s (avg ± sd) | llm.asm vs llama.cpp |"
  echo "|---|---|---|---|"
} > "$OUT_FILE"

# ---------- helpers: mean & sd ----------
stats() { # stdin = one float per line -> "mean sd" (2 decimals, 1 line); empty -> "n/a"
  awk '
    $1 !~ /^[0-9.]+$/ { next }
    { x[++n]=$1; s+=$1 }
    END {
      if (!n) { print "n/a"; exit }
      m=s/n
      if (n==1) { printf "%.2f %.2f", m, 0; exit }
      v=0; for(i=1;i<=n;i++) v+=(x[i]-m)^2
      printf "%.2f %.2f", m, sqrt(v/(n-1))
    }'
}

# strip " ± x" tail and keep the leading float in a markdown cell
cell_value() { # $1 = markdown row line
  sed -E 's/^[^|]*\|.*\|[^|]*\|[[:space:]]*([0-9]+([.][0-9]+)?)([[:space:]]*(±.*)?)?[[:space:]]*\|[[:space:]]*$/\1/' <<<"$1"
}

# ---------- run ----------
for t in "${TLIST[@]}"; do
  info "== threads=$t =="
  # --- llama.cpp (single invocation; -r reps produce avg ± sd) ---
  LB_TG=""
  SDV=""
  # Prefer machine-readable JSON when python3 is available; fall back to md
  HAVE_PY=0
  command -v python3 >/dev/null 2>&1 && HAVE_PY=1
  if [ "$HAVE_PY" -eq 1 ]; then
    LB_CMD=("$BENCH_EXEC" -m "$GGUF" -p "$PP" -n "$NG" -t "$t" -r "$REPS" -ngl "$NGL" -o json)
    info "\$ ${LB_CMD[*]}"
    if LB_JSON=$("${LB_CMD[@]}" 2>>"$RAW_ASME"); then
      LB_TG=$(printf '%s' "$LB_JSON" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
cand=[r for r in rows if r.get("n_gen")=='"$NG"' and r.get("n_prompt")==0]
if cand:
    r=cand[-1]
    print(r.get("avg_ts",""), r.get("stddev_ts",""))
') || LB_TG=""
    else
      info "llama-bench json failed for t=$t (see log); falling back to md"
    fi
  fi
  if [ -z "$LB_TG" ]; then
    LB_CMD=("$BENCH_EXEC" -m "$GGUF" -p "$PP" -n "$NG" -t "$t" -r "$REPS" -ngl "$NGL" -o md)
    info "\$ ${LB_CMD[*]}"
    if ! LB_MD=$("${LB_CMD[@]}" 2>>"$RAW_ASME"); then
      info "llama-bench FAILED for t=$t (see log)"; LB_TG="n/a"
    else
      while IFS= read -r row; do
        if grep -qE "\|[[:space:]]+tg${NG}[[:space:]]*\|" <<<"$row"; then
          LB_TG=$(cell_value "$row"); break
        fi
      done <<<"$LB_MD"
      [ -n "$LB_TG" ] || { info "could not parse tg tok/s from llama-bench (saved $RAW_ASME)"; LB_TG="n/a"; }
    fi
  fi
  # json path yields "avg stddev"; md path yields "avg"
  LB_TG=$(awk -v s="$LB_TG" 'BEGIN{split(s,a," "); if (a[1]!="") printf "%.2f", a[1]; if (a[2]!="") printf " %.2f", a[2]}')
  LB_VAL=${LB_TG%%[[:space:]]*}
  LB_SD=${LB_TG#*[[:space:]]}
  [ "$LB_SD" = "$LB_TG" ] && LB_SD=""
  # --- llm.asm (untimed warmup run, then one run per rep) ---
  AME_TPS=""
  AM=("$ASM_EXEC" -m "$ASM_MODEL" --tokens "$TOK_IDS" -n "$NG" -t "$t" -c $((PP+NG+64)))
  if [ "${NO_WARMUP:-0}" != "1" ]; then
    info "\$ ${AM[*]}  (warmup, not measured)"
    "${AM[@]}" >/dev/null 2>>"$RAW_ASME" || true
  fi
  for rep in $(seq "$REPS"); do
    info "\$ ${AM[*]}  (rep $rep/$REPS)"
    if ! "${AM[@]}" >>"$RAW_ASME" 2>>"$RAW_ASME"; then
      info "asm-llm FAILED for t=$t (see $RAW_ASME)"; break
    fi
    # perf lines (stdout+stderr both captured); latest decode line == this rep
    TPS=$(sed -nE 's/^\[asm-llm\] decode[[:space:]]*:[[:space:]]*[0-9]+ tok in [0-9.]+ ms \(([0-9.]+) tok\/s.*$/\1/p' "$RAW_ASME" | tail -1)
    [ -n "$TPS" ] || { info "could not parse asm decode tok/s (saved $RAW_ASME)"; break; }
    AME_TPS+="$TPS"$'\n'
  done
  ASME_STATS=$(stats <<<"${AME_TPS:-}")
  ASME_MEAN=${ASME_STATS%% *}
  ASME_SD=${ASME_STATS##* }
  if [ "$ASME_MEAN" = "n/a" ]; then AROW="failed"; else AROW="**${ASME_MEAN} ± ${ASME_SD}**"; fi

  # --- ratio ---
  if [ "$LB_VAL" != "n/a" ] && [ "$ASME_MEAN" != "n/a" ] && [ -n "$LB_VAL" ]; then
    RATIO=$(awk -v a="$ASME_MEAN" -v b="$LB_VAL" 'BEGIN{printf "%.2f (%+.1f%%)", a/b, (a/b-1)*100}')
  else
    RATIO="n/a"
  fi
  LROW="n/a"; AROW="${AROW:-n/a}"
  [ "$LB_VAL" != "n/a" ] && [ -n "$LB_VAL" ] && LROW="**$LB_VAL${LB_SD:+ ± $LB_SD}**"
  if [ "$LB_VAL" != "n/a" ] && [ "$ASME_MEAN" != "n/a" ]; then RATIO=$(awk -v a="$ASME_MEAN" -v b="$LB_VAL" 'BEGIN{printf "%.2f (%+.1f%%)", a/b, (a/b-1)*100}'); else RATIO="n/a"; fi
  echo "| $t | $LROW | $AROW | $RATIO |" >> "$OUT_FILE"
  info "threads=$t  llama.cpp=$LB_TG  llm.asm=$ASME_MEAN±$ASME_SD  => $RATIO"
done

info ""
info "== done — table appended to: $OUT_FILE =="
cat "$OUT_FILE"
