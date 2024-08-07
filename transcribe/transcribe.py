import argparse
import os
import threading
import time
from datetime import datetime, timedelta
from queue import Queue
from sys import platform

import numpy as np
import speech_recognition as sr
import torch
import whisper
from speech_recognition import WaitTimeoutError

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

    # internal variables
    non_speaking = [False]

    with source:
        # 주변 소음에 대한 인식기 감도를 조정하고 마이크에서 오디오를 녹음합니다.
        # 1초 동안 오디오 소스를 분석하기 때문에 1초 후부터 음성을 인식할 수 있다.
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        old_value = non_speaking[0]
        non_speaking[0] = True if audio is None else False
        if old_value != non_speaking[0]:
            print(f"non-speaking status : [{old_value} > {non_speaking[0]}]")

        if audio is not None:
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    listen_in_background(recorder, source, record_callback, phrase_time_limit=record_timeout)

    create_directory(LOG_DIR_NAME)
    log_path = os.path.join(LOG_DIR_NAME, make_filename('txt'))
    log_file = open(log_path, 'w')
    print(f">>> log file path: {log_path}")
    write_log_header(log_file)

    # Cue the user that we're ready to go.
    print(">>> Recorder is ready.\n")
    time.sleep(1)
    print(">>> START\n")

    just_first_time = True
    speech_in_progress = ''
    speech_timestamp = None

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

                new_speech_started = False
                if just_first_time or (phrase_complete and speech_in_progress):
                    just_first_time = False
                    new_speech_started = True
                else:
                    if not speech_in_progress:
                        # case: record finished로 이전 speech 완료된 상태
                        new_speech_started = True

                if new_speech_started:
                    # 새로운 speech 시작
                    print(f"{timestamp_format(phrase_time)} [new speech started]")
                    send_new_speech_started(phrase_time)

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
                end_time = time.perf_counter()
                timestamp_string = timestamp_format(phrase_time)
                elapsed_time_string = elapsed_time_format(start_time, end_time)
                print(f"{timestamp_string} [{elapsed_time_string}ms, phrase_complete: {phrase_complete}, "
                      f"speech_in_progress: {bool(speech_in_progress)}] {text}")

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise, edit the existing one.
                if phrase_complete and speech_in_progress:
                    # 진행 중이던 speech 완료 처리
                    speech_timestamp_string = timestamp_format(speech_timestamp)
                    print(f">>>{speech_timestamp_string} [phrase_complete] {speech_in_progress}")
                    send_stt(speech_timestamp_string, speech_in_progress)
                    write_stt(log_file, speech_timestamp_string, speech_in_progress)

                    # 새로운 speech 시작
                    speech_timestamp = phrase_time
                    speech_in_progress = text
                else:
                    if not speech_in_progress:
                        # record finished로 이전 speech 완료 후 새로운 speech 시작
                        speech_timestamp = phrase_time
                        speech_in_progress = text
                    else:
                        speech_in_progress += ' ' + text

                print(f"speech_in_progress={speech_in_progress}")

                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                # non-speaking 간주 기준: 1초 동안 listening 하여 0.8초(pause_threshold) 동안 말하지 않거나
                # 더 이상 오디오 입력이 없음.
                time.sleep(1)

                # 1초 후에 non-speaking 상태를 체크한다.
                if non_speaking[0] and speech_in_progress:
                    print("No speech detected. The phrase is considered complete.")
                    speech_timestamp_string = timestamp_format(speech_timestamp)
                    print(f">>>{speech_timestamp_string} [record_finished] {speech_in_progress}")
                    send_stt(speech_timestamp_string, speech_in_progress)
                    write_stt(log_file, speech_timestamp_string, speech_in_progress)
                    speech_in_progress = ''

        except KeyboardInterrupt:
            break

    log_file.close()
    print("\n\n>>> END")


def listen_in_background(recognizer, source, callback, phrase_time_limit=None):
    """
    소스 출처 : speech_recognition > Recognizer > listen_in_background()

    Spawns a thread to repeatedly record phrases from ``source`` (an ``AudioSource`` instance) into an
    ``AudioData`` instance and call ``callback`` with that ``AudioData`` instance as soon as each phrase are
    detected.

    Returns a function object that, when called, requests that the background listener thread stop. The
    background thread is a daemon and will not stop the program from exiting if there are no other non-daemon
    threads. The function accepts one parameter, ``wait_for_stop``: if truthy, the function will wait for the
    background listener to stop before returning, otherwise it will return immediately and the background
    listener thread might still be running for a second or two afterward. Additionally, if you are using a
    truthy value for ``wait_for_stop``, you must call the function from the same thread you originally called
    ``listen_in_background`` from.

    Phrase recognition uses the exact same mechanism as ``recognizer_instance.listen(source)``. The
    ``phrase_time_limit`` parameter works in the same way as the ``phrase_time_limit`` parameter for
    ``recognizer_instance.listen(source)``, as well.

    The ``callback`` parameter is a function that should accept two parameters - the ``recognizer_instance``,
    and an ``AudioData`` instance representing the captured audio. Note that ``callback`` function will be called
    from a non-main thread.
    """
    running = [True]

    def threaded_listen():
        with source as s:
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = recognizer.listen(s, 1, phrase_time_limit)
                except WaitTimeoutError:  # listening timed out, just try again
                    # modified: non-speaking 일 경우, audio data를 None으로 설정한다.
                    if running[0]:
                        callback(recognizer, None)
                    # pass
                else:
                    if running[0]:
                        callback(recognizer, audio)

    def stopper(wait_for_stop=True):
        running[0] = False
        if wait_for_stop:
            listener_thread.join()  # block until the background thread is done, which can take around 1 second

    listener_thread = threading.Thread(target=threaded_listen)
    listener_thread.daemon = True
    listener_thread.start()
    return stopper


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
    # | Timestamp (yyyy‑MM‑dd   | STT				                                          |
    # | HH:mm:ss.SSS)			|                                                             |
    # +---------------------------------------------------------------------------------------+
    file.write(f"+{'-' * 87}+\n")
    file.write(f"| Timestamp (yyyy‑MM‑dd{TAB_CHAR}| STT{TAB_CHAR * 8}|\n")
    file.write(f"| HH:mm:ss.SSS){TAB_CHAR * 2}|{TAB_CHAR * 8}|\n")
    file.write(f"+{'-' * 87}+\n")


def write_stt(file, timestamp, stt):
    # ex) 2024-07-26 13:44:01.748	 xxxxxxxx
    file.write(f"{timestamp}{TAB_CHAR} {stt}\n")


def send_new_speech_started(timestamp):
    # TODO: send a new speech starting point
    pass


def send_stt(timestamp_string, stt):
    # TODO: send stt
    pass


if __name__ == "__main__":
    main()
