#!/usr/bin/env bash
# Reconstruct a working hamer environment from a fresh checkout.
# Idempotent — safe to re-run; each step is skipped if already satisfied.
#
# Usage:
#     cd /path/to/hamer
#     bash scripts/rebuild_env.sh
#
# Overrides:
#     MANO_SOURCE=/path/to/MANO_RIGHT.pkl  bash scripts/rebuild_env.sh
#     TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121  bash scripts/rebuild_env.sh
#
# See docs/SUBMODULE_SETUP.md and docs/REBUILD_SCRIPT_PLAN.md.

set -euo pipefail

HAMER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HAMER_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
# Match the system CUDA toolkit. The reference setup uses torch 2.6 + cu124,
# which detectron2 builds cleanly against on driver >=535. Override with the
# TORCH_INDEX_URL env var if you need a different CUDA channel.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
MANO_SOURCE_DEFAULTS=(
    "${MANO_SOURCE:-}"
    "/misc/dl00/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl"
    "/home/gayagaya/ft-change/mano_v1_2/models/MANO_RIGHT.pkl"
)

log()  { printf "\033[1;34m[rebuild]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[rebuild]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[rebuild]\033[0m %s\n" "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. Python version
# --------------------------------------------------------------------------
log "step 1/12: checking python ($PYTHON_BIN)"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    die "$PYTHON_BIN not found. Install python3.10 or set PYTHON_BIN."
fi
"$PYTHON_BIN" --version

# --------------------------------------------------------------------------
# 2. Nested git submodules (ViTPose)
# --------------------------------------------------------------------------
log "step 2/12: nested submodules"
if [ -f third-party/ViTPose/setup.py ]; then
    log "  ViTPose already checked out, skipping"
else
    git submodule update --init --recursive
fi

# --------------------------------------------------------------------------
# 3. venv (prefer uv > virtualenv > python -m venv)
# --------------------------------------------------------------------------
log "step 3/12: venv (.hamer)"
if [ -x .hamer/bin/python ]; then
    log "  .hamer/ already exists, reusing"
elif command -v uv >/dev/null 2>&1; then
    log "  creating with uv (--seed for pip/setuptools/wheel)"
    uv venv --seed --python "$PYTHON_BIN" .hamer
elif command -v virtualenv >/dev/null 2>&1; then
    log "  creating with virtualenv"
    virtualenv -p "$PYTHON_BIN" .hamer
elif "$PYTHON_BIN" -c "import ensurepip" 2>/dev/null; then
    log "  creating with $PYTHON_BIN -m venv"
    "$PYTHON_BIN" -m venv .hamer
else
    die "no venv tool available. Install one of: uv, virtualenv, python3.10-venv (apt install python3.10-venv)"
fi

# shellcheck disable=SC1091
source .hamer/bin/activate
hash -r 2>/dev/null || true   # flush any stale pip/python from prior PATH
log "  using $(python --version) at $(which python)"
log "  pip resolves to $(which pip)"

# --------------------------------------------------------------------------
# 4. pip upgrade + pin numpy<2 (xtcocotools C ext is built against
#    numpy 1.x ABI; the reference venv uses 1.26.4)
# --------------------------------------------------------------------------
log "step 4/12: pip upgrade + numpy<2"
pip install --upgrade pip >/dev/null
pip install 'numpy<2' >/dev/null

# --------------------------------------------------------------------------
# 5. PyTorch
# --------------------------------------------------------------------------
log "step 5/12: torch (index: $TORCH_INDEX_URL)"
if python -c "import torch" 2>/dev/null; then
    log "  torch already installed ($(python -c 'import torch;print(torch.__version__)'))"
else
    pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

# --------------------------------------------------------------------------
# 6. detectron2 (built separately with --no-build-isolation: it imports
#    torch in setup.py, which PEP 517 isolated builds don't expose)
# --------------------------------------------------------------------------
log "step 6/12: detectron2 (no-build-isolation, this can take 10-20 min)"
if python -c "import detectron2" 2>/dev/null; then
    log "  detectron2 already installed"
else
    # build deps detectron2's setup.py expects.
    # Pin setuptools<70: setuptools>=81 drops pkg_resources which torch's
    # cpp_extension still imports.
    pip install 'setuptools<70' wheel ninja
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2'
fi

# --------------------------------------------------------------------------
# 7. hamer + extras (detectron2 already satisfied, so this is quick).
#    Check a key runtime dep (smplx) in addition to `hamer` itself, because
#    earlier failed runs can leave the egg-link in place without the deps.
# --------------------------------------------------------------------------
log "step 7/12: hamer + extras"
if python -c "import hamer, smplx, pytorch_lightning, mmcv" 2>/dev/null; then
    log "  hamer + core deps already importable"
else
    # setup.py declares detectron2 as a git+URL dep, which pip re-clones
    # every time. --no-build-isolation lets that re-build succeed if it
    # happens; --upgrade-strategy only-if-needed avoids re-fetching deps
    # whose installed version already satisfies the spec.
    # Pass the numpy<2 constraint inline so transitive deps don't drag in
    # numpy 2.x and break xtcocotools' C ext (built against 1.x ABI).
    pip install --no-build-isolation --upgrade-strategy only-if-needed \
        -e ".[all]" 'numpy<2'
fi

# --------------------------------------------------------------------------
# 8. ViTPose
# --------------------------------------------------------------------------
log "step 8/12: ViTPose"
if python -c "import sys, pathlib; sys.path.insert(0, '$HAMER_ROOT'); import vitpose_model" 2>/dev/null; then
    log "  vitpose_model importable, skipping ViTPose install"
else
    pip install --no-build-isolation -v -e third-party/ViTPose
fi

# --------------------------------------------------------------------------
# 9. HaMeR checkpoints (fetch_demo_data.sh ≈ 6 GB)
# --------------------------------------------------------------------------
log "step 9/12: HaMeR checkpoints"
if [ -d _DATA/hamer_ckpts ] && find _DATA/hamer_ckpts -name "*.ckpt" | grep -q .; then
    log "  _DATA/hamer_ckpts/ already populated"
else
    log "  running fetch_demo_data.sh (~6 GB download)"
    bash fetch_demo_data.sh
fi
# fetch_demo_data.sh drops the tarball in cwd, but hamer.models.download_models
# expects it at _DATA/hamer_demo_data.tar.gz. Move it so subsequent calls
# don't re-download from the web.
if [ -f hamer_demo_data.tar.gz ] && [ ! -f _DATA/hamer_demo_data.tar.gz ]; then
    log "  moving hamer_demo_data.tar.gz into _DATA/"
    mv hamer_demo_data.tar.gz _DATA/
fi

# --------------------------------------------------------------------------
# 10. ViTPose checkpoint sanity
# --------------------------------------------------------------------------
log "step 10/12: ViTPose checkpoint"
if [ -d _DATA/vitpose_ckpts ] && find _DATA/vitpose_ckpts -name "*.pth" | grep -q .; then
    log "  _DATA/vitpose_ckpts/ ok"
else
    warn "  _DATA/vitpose_ckpts/ missing or empty — should have come from fetch_demo_data.sh; check the tarball"
fi

# --------------------------------------------------------------------------
# 11. MANO model (symlink from known location)
# --------------------------------------------------------------------------
log "step 11/12: MANO model"
MANO_DST="_DATA/data/mano/MANO_RIGHT.pkl"
if [ -e "$MANO_DST" ]; then
    log "  $MANO_DST already in place"
else
    mkdir -p "$(dirname "$MANO_DST")"
    found=""
    for cand in "${MANO_SOURCE_DEFAULTS[@]}"; do
        [ -z "$cand" ] && continue
        if [ -f "$cand" ]; then
            ln -s "$cand" "$MANO_DST"
            log "  symlinked from $cand"
            found="yes"
            break
        fi
    done
    if [ -z "$found" ]; then
        die "MANO_RIGHT.pkl not found. Set MANO_SOURCE=... or download from https://mano.is.tue.mpg.de and place at $MANO_DST"
    fi
fi

# --------------------------------------------------------------------------
# 12. Smoke test
# --------------------------------------------------------------------------
log "step 12/12: smoke test (loading pipeline — first run downloads model weights, slow)"
python - <<'PY'
import os, sys
HERE = os.path.dirname(os.path.abspath(os.getcwd()))
# build_pipeline cd's into hamer root itself, so we just need this module importable
from hamer_api import Hamer
h = Hamer(body_detector='regnety')
print(f"OK: pipeline loaded ({type(h.pipe['model']).__name__} on {h.pipe['device']})")
PY

log "DONE: hamer environment is ready. Activate with: source .hamer/bin/activate"
