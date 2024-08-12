#!/bin/bash

BASE_DIR="/home/obigo/github/whisper-rt"
RESULT="/home/obigo/github/whisper-rt/output/auto_runs.txt"

StartTime=$(date +%s)

get_time() {
	TIME=$(date '+%Y/%m/%d %H:%M:%S')
}

msg() {
	get_time
	echo "[$TIME] $*" | tee -a "$RESULT"
}

conda_act() {
	source /home/obigo/anaconda3/etc/profile.d/conda.sh
	conda activate whisper_rt
	conda info | tee -a "$RESULT"
}

msg "[START   ] Script Started!"

conda_act

cd $BASE_DIR || exit

# -u: unbuffered output
/home/obigo/anaconda3/envs/whisper_rt/bin/python -u /home/obigo/github/whisper-rt/main.py --model "large-v3"

EndTime=$(date +%s)

msg "[END     ] Script End!"
msg "[RUN-TIME] Total Spent time : $((EndTime - StartTime)) Second"
echo -e "\n" | tee -a "$RESULT"
