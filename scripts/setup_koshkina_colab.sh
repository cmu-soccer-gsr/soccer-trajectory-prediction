#!/bin/bash
# Set up Koshkina & Elder's jersey pipeline on Colab (once per session, ~20-30 min).
#
# Their code calls separate conda envs with `conda run -n <env>`:
#   vitpose   pose estimation (torch 1.9 + mmcv-full 1.4.8)
#   parseq2   jersey digit reader (PARSeq)
#   centroids ReID outlier filter (only if WITH_REID=1)
# Everything else (legibility classifier, voting) runs in Colab's own Python.
# Conda goes in its own folder, so Colab's Python and the sn-gamestate .venv are untouched.
#
# Usage: bash scripts/setup_koshkina_colab.sh        (WITH_REID=1 bash ... to add the ReID env)
set -e
KOSH_DIR=${KOSH_DIR:-/content/jersey-number-pipeline}
CONDA_DIR=${CONDA_DIR:-/opt/conda}
WITH_REID=${WITH_REID:-0}

# 1. Conda (Miniforge)
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "== Installing Miniforge"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR" > /dev/null
fi
export PATH="$CONDA_DIR/bin:$PATH"
has_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# 2. Their repo, plus the SAM optimizer their legibility code imports
[ -d "$KOSH_DIR/.git" ] || git clone -q --depth 1 https://github.com/mkoshkina/jersey-number-pipeline "$KOSH_DIR"
cd "$KOSH_DIR"
[ -d sam ] || git clone -q --depth 1 https://github.com/davda54/sam sam
[ -d pose/ViTPose ] || git clone -q --depth 1 https://github.com/ViTAE-Transformer/ViTPose pose/ViTPose

# 3. Colab's Python: only gdown is missing (torch, opencv, pandas, scipy are preinstalled)
pip install -q gdown

# 4. Weights: SoccerNet legibility classifier, SoccerNet PARSeq, ViTPose-H
mkdir -p models pose/ViTPose/checkpoints
python3 - <<'PY'
import os, gdown, configuration as c
sn = c.dataset["SoccerNet"]
for path, url in [(sn["legibility_model"], sn["legibility_model_url"]),
                  (sn["str_model"], sn["str_model_url"]),
                  ("pose/ViTPose/checkpoints/vitpose-h.pth", sn["pose_model_url"])]:
    if not os.path.isfile(path):
        gdown.download(url, path, quiet=True)
    print("ok" if os.path.isfile(path) else "MISSING", path)
PY

# 5. vitpose env. Their setup.py never installs torch here, so we pin it to match mmcv-full 1.4.8.
if ! has_env vitpose; then
  echo "== Creating vitpose env"
  conda create -y -q -n vitpose python=3.8 > /dev/null
  conda run -n vitpose pip install -q torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
      -f https://download.pytorch.org/whl/torch_stable.html
  conda run -n vitpose pip install -q mmcv-full==1.4.8 \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
  conda run -n vitpose pip install -q -e pose/ViTPose
  conda run -n vitpose pip install -q timm==0.4.9 einops "numpy<1.24"
fi

# 6. parseq2 env (their modified PARSeq is already in str/parseq)
if ! has_env parseq2; then
  echo "== Creating parseq2 env"
  conda create -y -q -n parseq2 python=3.9 > /dev/null
  (cd str/parseq && sed 's|cpu|cu117|' requirements/core.txt > requirements/core.cu117.txt \
     && conda run -n parseq2 pip install -q -r requirements/core.cu117.txt -e ".[train,test]")
fi

# 7. Optional: centroids env for the ReID outlier filter (their own setup function)
if [ "$WITH_REID" = 1 ] && ! has_env centroids; then
  echo "== Creating centroids env"
  python3 -c "import os, setup; setup.setup_reid(os.getcwd())"
fi

# 8. Check each env imports what it needs
echo "== Checks"
python3 -c "import torch, cv2, sam.sam; print('main     torch', torch.__version__, '| gpu', torch.cuda.is_available())"
conda run -n vitpose python -c "import torch, mmcv, mmpose; print('vitpose  torch', torch.__version__, '| gpu', torch.cuda.is_available(), '| mmcv', mmcv.__version__)"
conda run -n parseq2 python -c "import torch, strhub; print('parseq2  torch', torch.__version__, '| gpu', torch.cuda.is_available())"
if [ "$WITH_REID" = 1 ]; then conda run -n centroids python -c "import torch; print('centroids torch', torch.__version__)"; fi
echo "Setup done."
