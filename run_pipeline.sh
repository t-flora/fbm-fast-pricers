#!/usr/bin/env bash
# run_pipeline.sh — full reproducibility pipeline
#
# Builds C++ binaries, runs benchmark, runs all Python analysis scripts,
# then collects every output into a timestamped snapshot directory so
# each run is independently identifiable.
#
# Usage:
#   ./run_pipeline.sh            # full run (production parameters)
#   ./run_pipeline.sh --fast     # reduced M for quick smoke-test
#   ./run_pipeline.sh --no-build # skip cmake (binaries already built)
#   ./run_pipeline.sh --no-iv    # skip validate_iv.py (needs internet)
#
# Output layout:
#   plots/runs/YYYYMMDD_HHMMSS/
#     manifest.txt          — parameters, git commit, timing
#     *.png                 — copies of every generated plot
#     results/              — copies of benchmark CSVs + reference_price.txt

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────────────────────
FAST=0
NO_BUILD=0
NO_IV=0
for arg in "$@"; do
    case "$arg" in
        --fast)     FAST=1 ;;
        --no-build) NO_BUILD=1 ;;
        --no-iv)    NO_IV=1 ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Parameters (production vs fast) ──────────────────────────────────────────
if [ "$FAST" -eq 1 ]; then
    M_ASIAN=1000;  N_ASIAN=63
    M_IV=500;      N_IV=30
    M_SENS=1000;   N_SENS=63
    CONV_SEEDS=3;  CONV_MAX_M=5000
    STRUCT_SMALL=32; STRUCT_LARGE=64
    echo "Mode: FAST (reduced M for smoke-test)"
else
    M_ASIAN=10000; N_ASIAN=252
    M_IV=3000;     N_IV=63
    M_SENS=10000;  N_SENS=252
    CONV_SEEDS=5;  CONV_MAX_M=25000
    STRUCT_SMALL=64; STRUCT_LARGE=128
    echo "Mode: PRODUCTION"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="plots/runs/${TIMESTAMP}"
START_TIME=$(date +%s)

echo "══════════════════════════════════════════════════════════"
echo "  Pipeline run: ${TIMESTAMP}"
echo "  Snapshot dir: ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════"

# ── Step 1: Build ─────────────────────────────────────────────────────────────
if [ "$NO_BUILD" -eq 0 ]; then
    echo ""
    echo "── Step 1/10: cmake build ──"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev --log-level=WARNING
    cmake --build build --parallel
    echo "  done."
else
    echo ""
    echo "── Step 1/10: cmake build — SKIPPED (--no-build) ──"
fi

# ── Step 2: C++ benchmark ─────────────────────────────────────────────────────
echo ""
echo "── Step 2/10: C++ benchmark ──"
./build/benchmark

# ── Step 3: Scaling + construction breakdown + memory plots ───────────────────
echo ""
echo "── Step 3/10: plot_scaling.py ──"
uv run python plots/plot_scaling.py

# ── Step 4: Python memory profiling ───────────────────────────────────────────
echo ""
echo "── Step 4/10: profile_memory.py ──"
uv run python data/profile_memory.py

# ── Step 5: Structural analysis ───────────────────────────────────────────────
echo ""
echo "── Step 5/10: plot_structure.py (N-small=${STRUCT_SMALL}, N-large=${STRUCT_LARGE}) ──"
uv run python plots/plot_structure.py --N-small "${STRUCT_SMALL}" --N-large "${STRUCT_LARGE}"

# ── Step 6: MC convergence ────────────────────────────────────────────────────
echo ""
echo "── Step 6/10: validate_convergence.py (seeds=${CONV_SEEDS}, max-M=${CONV_MAX_M}) ──"
uv run python data/validate_convergence.py --n-seeds "${CONV_SEEDS}" --max-M "${CONV_MAX_M}"

# ── Step 7: Stability ─────────────────────────────────────────────────────────
echo ""
echo "── Step 7/10: validate_stability.py ──"
uv run python data/validate_stability.py

# ── Step 8: Lévy benchmark + roughness premium ────────────────────────────────
echo ""
echo "── Step 8/10: validate_asian.py (M=${M_ASIAN}, N=${N_ASIAN}) ──"
uv run python data/validate_asian.py --M "${M_ASIAN}" --N "${N_ASIAN}"

# ── Step 9: IV smile vs SPY (requires internet) ───────────────────────────────
echo ""
if [ "$NO_IV" -eq 0 ]; then
    echo "── Step 9/10: validate_iv.py (M=${M_IV}, N=${N_IV}) ──"
    uv run python data/validate_iv.py --M "${M_IV}" --N "${N_IV}" || {
        echo "  WARNING: validate_iv.py failed (network or data issue) — continuing."
    }
else
    echo "── Step 9/10: validate_iv.py — SKIPPED (--no-iv) ──"
fi

# ── Step 10: Sensitivity heatmap + price vs strike ────────────────────────────
echo ""
echo "── Step 10/10: plot_sensitivity.py (M=${M_SENS}, N=${N_SENS}) ──"
uv run python plots/plot_sensitivity.py --M "${M_SENS}" --N "${N_SENS}"

# ── Collect outputs ───────────────────────────────────────────────────────────
echo ""
echo "── Collecting outputs → ${RUN_DIR} ──"
mkdir -p "${RUN_DIR}/results"

# Plots
cp plots/figures/*.png "${RUN_DIR}/" 2>/dev/null || true

# Benchmark CSVs and reference price
cp benchmarks/results/*.csv  "${RUN_DIR}/results/" 2>/dev/null || true
cp benchmarks/results/*.txt  "${RUN_DIR}/results/" 2>/dev/null || true

# ── Write manifest ────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "n/a")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "n/a")
GIT_STATUS=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

cat > "${RUN_DIR}/manifest.txt" << EOF
════════════════════════════════════════
Run ID    : ${TIMESTAMP}
Date      : $(date)
Wall time : ${ELAPSED}s
════════════════════════════════════════
Git
  branch  : ${GIT_BRANCH}
  commit  : ${GIT_COMMIT}
  dirty   : ${GIT_STATUS} uncommitted file(s)

Mode      : $([ "$FAST" -eq 1 ] && echo "FAST" || echo "PRODUCTION")

Parameters
  validate_asian      --M ${M_ASIAN} --N ${N_ASIAN}
  validate_iv         --M ${M_IV}    --N ${N_IV}  $([ "$NO_IV" -eq 1 ] && echo "(SKIPPED)" || echo "")
  plot_sensitivity    --M ${M_SENS}  --N ${N_SENS}
  validate_convergence --n-seeds ${CONV_SEEDS} --max-M ${CONV_MAX_M}
  plot_structure      --N-small ${STRUCT_SMALL} --N-large ${STRUCT_LARGE}
  validate_stability  (no flags)
  profile_memory      (no flags)       → plots/memory_profile.png

Plots
$(ls "${RUN_DIR}"/*.png 2>/dev/null | xargs -n1 basename | sed 's/^/  /')

Benchmark results
$(ls "${RUN_DIR}/results/" 2>/dev/null | sed 's/^/  /')
════════════════════════════════════════
EOF

# ── Summary ───────────────────────────────────────────────────────────────────
N_PLOTS=$(ls "${RUN_DIR}"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done in ${ELAPSED}s."
echo "  ${N_PLOTS} plots + CSV results → ${RUN_DIR}"
echo "  Manifest: ${RUN_DIR}/manifest.txt"
echo "══════════════════════════════════════════════════════════"
