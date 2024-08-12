#!/bin/sh

BASE_DIR="/파일이 있는 경로를 입력"
RESULT="/실행 결과를 출력할 경로를 입력"

StartTime=$(date +%s)

get_time() {
	TIME=$(date '+%Y/%m/%d %H:%M:%S')
}

msg() {
	get_time
	echo "[$TIME] $*" | tee -a "$RESULT"
}

conda_act() {
	source ~anaconda3/etc/profile.d/conda.sh
	conda activate whisper_rt
}

msg "[START   ] Script Started!"

conda_act

cd $BASE_DIR

~/anaconda3/envs/가상환경 이름/bin/python `파일명(~.py)`

EndTime=$(date +%s)

msg "[END     ] Script End!"
msg "[RUN-TIME] Total Spent time : $(($EndTime - $StartTime)) Second"