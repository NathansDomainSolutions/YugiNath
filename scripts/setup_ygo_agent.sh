#!/usr/bin/env bash
# Install ygo-agent (duel engine + PPO trainer) into external/, with its own venv.
#
# Needs: Ubuntu 22.04+ (or WSL2), git, curl, make, and Python 3.10, 3.11 or 3.12
# (ygo-agent ships prebuilt engine binaries only for those). Pick the Python with
#   PYTHON=python3.11 scripts/setup_ygo_agent.sh
# Add CUDA=1 to install the GPU build of JAX (NVIDIA GPU + CUDA 12 driver).
#
# Version pins below were chosen to match ygo-agent's "jax<=0.4.28" requirement;
# they have not yet been tested together on a real machine. If pip reports a
# conflict, open an issue with the error.
set -euo pipefail

YGO_AGENT_COMMIT=dbf5142d49aab2e6beb4150788d4fffec39ae3e5   # main as of 2024-08-16
PYTHON=${PYTHON:-python3.12}

cd "$(dirname "$0")/.."
mkdir -p external

if [ ! -d external/venv ]; then
  "$PYTHON" -m venv external/venv
fi
# shellcheck disable=SC1091
source external/venv/bin/activate
python -m pip install -U pip wheel

if [ "${CUDA:-0}" = "1" ]; then
  pip install "jax[cuda12]<=0.4.28" "jaxlib<=0.4.28"
else
  pip install "jax<=0.4.28" "jaxlib<=0.4.28"
fi
pip install "numpy<2" "flax==0.8.5" "optax==0.2.2" "chex==0.1.86" "distrax==0.1.5" \
  "orbax-checkpoint==0.5.20" tyro pandas tensorboardX tqdm

if [ ! -d external/ygo-agent ]; then
  git clone https://github.com/sbl1996/ygo-agent.git external/ygo-agent
fi
cd external/ygo-agent
git fetch --depth 1 origin "$YGO_AGENT_COMMIT" 2>/dev/null || true
git checkout -q "$YGO_AGENT_COMMIT"

PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
SO=ygoenv/ygoenv/ygopro/ygopro_ygoenv.so
if [ ! -f "$SO" ]; then
  curl -fL -o "$SO" "https://github.com/sbl1996/ygo-agent/releases/download/v0.1/ygopro_ygoenv_${PYTAG}.so"
fi

# Downloads the card database (mycard/ygopro-database) and card scripts
# (mycard/ygopro-scripts), then pip-installs ygoenv, ygoinf and ygoai.
make

cd scripts
echo "== Checking duels work (random play, 32 games) =="
python -u eval.py --deck ../assets/deck/ --num_episodes 32 --strategy random --num_envs 16

echo
echo "ygo-agent is ready. From the repo root, try:"
echo "  external/venv/bin/python -m yuginath --config config/smoke.toml run"
