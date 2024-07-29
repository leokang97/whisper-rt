# Whisper Real Time ASR (Automatic Speech Recognition)

OpenAI Whisper transformers model을 사용하여 실시간 음성인식하는 파이썬 앱

<br>

## Purpose
마이크로 입력되는 사용자의 음성을 지속적으로 음성인식하여 STT를 결과로 만드는 것이 목적이다.

<br>

## Pre-requisites
- Target : Linux OS, Ubuntu 20.04 이상 (22.04 권장)
- Python : 3.9 이상 (3.10 권장)
- requirements.txt 참고

<br>

## Getting Started
- conda 환경을 생성하여 실행하는 것을 권장함
1. 타겟 시스템에 알맞는 torch, cuda 환경 설치
2. ffmpeg 설치
```
$ conda install ffmpeg
```
(참고 : How to use Whisper in Python
https://nicobytes.com/blog/en/how-to-use-whisper/)

3. whisper-rt requirements
```
$ conda install pyaudio
$ conda install SpeechRecognition
```
4. whisper 설치
```
$ pip install -U openai-whisper
```
(참고 : pip 명령어의 경우 conda, conda-forge로 안되는 패키지를 설치할 때만 사용한다.
https://biomadscientist.tistory.com/114)

<br>

## Usage
"whisper-rt" 패키지 상위 디렉토리에서 아래의 파이썬 스크립트를 실행한다.
```
$ python whisper-rt --model "large-v3"
or
$ python whisper-rt --model "medium" --non_english
```

<br>

## Support
- TBD

<br>

## License

Copyright 2022 OBIGO Inc. all right reserved.

This software is covered by the license agreement between
the end user and OBIGO Inc. and may be
used and copied only in accordance with the terms of the
said agreement.

OBIGO Inc. assumes no responsibility or
liability for any errors or inaccuracies in this software,
or any consequential, incidental or indirect damage arising
out of the use of the software.

