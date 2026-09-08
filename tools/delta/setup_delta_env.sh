#!/bin/bash
# Reproducible NCSA Delta environment using the CUDA 12.8 PyTorch wheel.
set -euo pipefail

VENV_PATH="${VENV_PATH:?Set VENV_PATH to a project-storage virtual environment path}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"

module reset
module load "${PYTHON_MODULE}"

python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip

# Install this first. requirements.txt accepts this version and therefore will
# not replace it with the incompatible default CUDA 13 wheel.
python -m pip install \
  "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python -c 'import torch; print(f"torch={torch.__version__} cuda_build={torch.version.cuda}"); assert torch.__version__.startswith("2.11.0") and torch.version.cuda == "12.8", "Delta requires the pinned PyTorch CUDA 12.8 build"'
echo "Delta environment ready: ${VENV_PATH}"
