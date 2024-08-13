#!/bin/bash
source /home/obigo/anaconda3/etc/profile.d/conda.sh
conda activate whisper_rt

# -u: unbuffered output
python -u /home/obigo/github/whisper-rt/main.py --model "large-v3"
