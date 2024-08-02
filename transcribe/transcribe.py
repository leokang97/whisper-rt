import argparse
import os
import time
from datetime import datetime, timedelta
from queue import Queue
from sys import platform

import numpy as np
import speech_recognition as sr
import torch
import whisper

# references
# https://github.com/davabase/whisper_real_time

TAB_CHAR = '\t'
SAMPLE_RATE = 16_000
LOG_DIR_NAME = 'output'


def main():
    parser = argparse.ArgumentParser()
    # models: (tiny.en, tiny), (base.en, base), (small.en, small), (medium.en, medium),
    # (large-v1, large-v2, large-v3, large) 
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large-v3", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=3,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # dynamic energy compensation이 SpeechRecognizer가 녹음을 멈추지 않는 지점까지 energy threshold 값을 극적으로 낮춘다.
    recorder.dynamic_energy_threshold = False
    # parameters

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"[{index}] Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=SAMPLE_RATE, device_index=index)
                    print(f">>> Selected Microphone name : \"{name}\"\n")
                    break
    else:
        source = sr.Microphone(sample_rate=SAMPLE_RATE)

    print(f">>> cuda available: {torch.cuda.is_available()}")
    print(f">>> mps available: {torch.backends.mps.is_available()}")

    # Load / Download model
    model = args.model
    if not args.model.startswith('large') and not args.non_english:
        model = model + ".en"
    # Parameters:
    # in_memory (bool): whether to preload the model weights into host memory
    # notes: Apple Silicon(M1, M2 칩)에서 device='mps' 형식은 지원되지 않는다.
    # NotImplementedError: Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with arguments
    # from the 'SparseMPS' backend
    audio_model = whisper.load_model(name=model)
    print(f">>> \"{model}\" Model loaded.\n")

    # How real time the recording is in seconds.
    record_timeout = args.record_timeout
    # How much empty space between recordings before we consider it a new line in the transcription. (unit: seconds)
    phrase_timeout = args.phrase_timeout

    only_once = True

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    create_directory(LOG_DIR_NAME)
    log_path = os.path.join(LOG_DIR_NAME, make_filename('txt'))
    log_file = open(log_path, 'w')
    print(f">>> log file path: {log_path}")
    write_log_header(log_file)

    # Cue the user that we're ready to go.
    print(">>> Recorder is ready.\n")
    time.sleep(0.5)
    print(">>> START\n")

    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # 녹음 사이에 충분한 시간이 경과한 경우에는 문구가 완료되었다고 간주한다.
                # 새 데이터로 다시 시작하려면 현재 작동 중인 오디오 버퍼를 지운다.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                start_time = time.perf_counter()

                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # in-ram buffer를 임시파일 없이 모델이 직접 사용할 수 있는 것으로 변환한다.
                # 데이터를 16 bit wide integers에서 floating point with a width of 32 bits로 변환한다. 
                # audio stream frequency를 최대 32,768hz의 PCM 파장 호환 기본값으로 고정합니다.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                # Parameters:
                # language (str): language spoken in the audio, specify None to perform language detection
                #     (default: None)
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language='ko')
                text = result['text'].strip()
                if not text:
                    continue
                end_time = time.perf_counter()

                # TODO: STT 전달
                # send(text)
                new_sentence = True if only_once or phrase_complete else False
                only_once = False
                timestamp_string = timestamp_format(phrase_time)
                elapsed_time_string = elapsed_time_format(start_time, end_time)
                new_sentence_string = 'O' if new_sentence else 'X'
                print(f"{timestamp_string} [{elapsed_time_string}ms, new: {new_sentence_string}] {text}")
                write_stt(log_file, timestamp_string, elapsed_time_string, new_sentence_string, text)

                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                time.sleep(0.25)
        except KeyboardInterrupt:
            break

    log_file.close()
    print("\n\n>>> END")


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def make_filename(ext):
    dt = datetime.now().strftime('%y%m%d_%H%M%S')
    return 'transcribe_' + dt + '.' + ext


def timestamp_format(dt: datetime):
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def elapsed_time_format(start, end):
    return int(round((end - start) * 1000))


def write_log_header(file):
    # log header
    # +---------------------------------------------------------------------------------------+
    # | Timestamp (yyyy‑MM‑dd   | Elapsed	| New		| STT				                  |
    # | HH:mm:ss.SSS)			| Time (ms)	| Sentence	|				                      |
    # +---------------------------------------------------------------------------------------+
    file.write(f"+{'-' * 87}+\n")
    file.write(f"| Timestamp (yyyy‑MM‑dd{TAB_CHAR}| Elapsed{TAB_CHAR}| New{TAB_CHAR * 2}| STT{TAB_CHAR * 4}|\n")
    file.write(f"| HH:mm:ss.SSS){TAB_CHAR * 2}| Time (ms){TAB_CHAR}| Sentence{TAB_CHAR}|{TAB_CHAR * 4}|\n")
    file.write(f"+{'-' * 87}+\n")


def write_stt(file, timestamp, elapsed_time, new_sentence, stt):
    # ex) 2024-07-26 13:44:01.748	 [129ms]	 [O]		 xxxxxxxx
    file.write(f"{timestamp}{TAB_CHAR} [{elapsed_time}ms]{TAB_CHAR} [{new_sentence}]{TAB_CHAR * 2} {stt}\n")


if __name__ == "__main__":
    main()
