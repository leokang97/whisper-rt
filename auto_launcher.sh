#!/bin/bash
source "$HOME"/anaconda3/etc/profile.d/conda.sh
conda activate whisper_rt

# -u: unbuffered output
python -u "$HOME"/github/whisper-rt/main.py --model "large-v3"
