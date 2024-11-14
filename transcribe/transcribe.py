import argparse
import audioop
import json
import logging
import re
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

import constants
from proto.asr_client import AsrClient
from utils import file_util
from utils.timer_util import MyTimer

# references
# https://github.com/davabase/whisper_real_time

SAMPLE_RATE = 16_000
dummy_stt = ['감사합니다.', '시청해주셔서 감사합니다.']
wuw_pattern = re.compile(r'(하이|헤이|해이)+\s?(젠|겐|렌)+')

logger = logging.getLogger('transcribe')
logger.setLevel(logging.DEBUG)

# console logger
console_formatter = logging.Formatter(constants.LOG_FORMATTER)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# file logger
file_formatter = logging.Formatter(constants.LOG_FORMATTER)
log_path = '%s/%s' % (constants.LOG_DIR_NAME, file_util.make_filename('transcribe', 'log'))
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


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
    parser.add_argument("--mic_debiased", action='store_true',
                        help="Whether or not to check debiased energy to check microphone working.")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    mic_source = None
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # dynamic energy compensation이 SpeechRecognizer가 녹음을 멈추지 않는 지점까지 energy threshold 값을 극적으로 낮춘다.
    recorder.dynamic_energy_threshold = False
    # gRPC ASR client instance
    asr_client = AsrClient()
    once_asr_timer = MyTimer()

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            logger.debug("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                logger.debug(f"[{index}] Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    mic_source = sr.Microphone(sample_rate=SAMPLE_RATE, device_index=index)
                    logger.info(f"Selected Microphone name : \"{name}\"\n")
                    break
    else:
        mic_source = sr.Microphone(sample_rate=SAMPLE_RATE)

    logger.info(f"cuda available: {torch.cuda.is_available()}, mps available: {torch.backends.mps.is_available()}")

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
    logger.info(f"\"{model}\" Model loaded.\n")

    # How real time the recording is in seconds.
    record_timeout = args.record_timeout
    # How much empty space between recordings before we consider it a new line in the transcription. (unit: seconds)
    phrase_timeout = args.phrase_timeout

    # internal variables
    non_speaking = [False]
    stop_recognize = [False]  # PAD(android tablet)로부터 start/stop recognize 요청
    mouth_opened = [False]  # 얼굴인식 카메라로부터 mouth open state, true=opened, false=closed, default=closed
    soft_asr_blocking = [False]  # 물리적 마이크 버튼 ASR 차단이 아니라 소프트웨어적인 ASR 차단 요청
    force_start_recognize = [False]  # start recognize 요청 시 mouth state를 고려하지 않고 강제로 ASR 한다.
    once_start_recognize = [False]  # start recognize 요청 시 once [n] 초동안 mouth state를 고려하지 않고 ASR 한다.
    speech_in_progress = ['']

    assert isinstance(mic_source, sr.AudioSource), "Source must be an audio source"

    with mic_source:
        # 주변 소음에 대한 인식기 감도를 조정하고 마이크에서 오디오를 녹음합니다.
        # 1초 동안 오디오 소스를 분석하기 때문에 1초 후부터 음성을 인식할 수 있다.
        recorder.adjust_for_ambient_noise(mic_source)
    time.sleep(1)

    # notes: 조명 모듈에서 오디오 입력 사용하는 부분과 충돌이 있어서 기본값은 "사용 안함"
    # mic_debiased flag 기본값은 False
    # 필요할 시에 실행 arguments에 "--mic_debiased"을 추가하여 사용할 수 있음
    if 'linux' in platform and args.mic_debiased:
        if not check_microphone_working(args.default_microphone):
            logger.info("마이크가 동작하는지 마이크 전원을 확인하세요.")
            return

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        if audio is not None:
            logger.debug(f"record_callback, audio=True,soft_asr_blocking={soft_asr_blocking[0]}")

        if not soft_asr_blocking[0]:
            old_value = non_speaking[0]
            non_speaking[0] = True if audio is None else False
            if old_value != non_speaking[0]:
                logger.debug(f"non-speaking status : ({old_value} > {non_speaking[0]})")

            if audio is not None:
                # Grab the raw bytes and push it into the thread safe queue.
                data = audio.get_raw_data()
                data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    listen_in_background(recorder, mic_source, record_callback, phrase_time_limit=record_timeout)

    def once_asr_timer_finish_callback() -> None:
        once_start_recognize[0] = False
        logger.debug("once asr timer finish callback: once=False")

    class RpcEventCallback:
        @staticmethod
        def on_msg_control(data: str):
            data_obj = json.loads(data)
            logger.debug(f"gRPC event callback: on_msg_control, data={data_obj}")
            event_name = data_obj['event']
            old_value = soft_asr_blocking[0]
            if event_name == 'startRecognize':
                stop_recognize[0] = False

                # 'force' optional field
                force_field = data_obj.get('force')
                if force_field:
                    if force_field == 'true':
                        force_start_recognize[0] = True
                    elif force_field == 'false':
                        force_start_recognize[0] = False

                # 'once' optional field
                once_field = data_obj.get('once')

                logger.debug(f"gRPC event callback: on_msg_control, force={force_start_recognize[0]},once={once_field}sec.")
                if force_start_recognize[0]:
                    # force start ASR
                    soft_asr_blocking[0] = False
                elif once_field:
                    once_start_recognize[0] = True
                    once_asr_timer.start_timer(once_field, once_asr_timer_finish_callback)
                    # once start ASR
                    soft_asr_blocking[0] = False
                else:
                    # mouth state 고려함
                    soft_asr_blocking[0] = False if mouth_opened[0] else True
                send_asr_state_changed(asr_client, old_value, soft_asr_blocking[0])
            elif event_name == 'stopRecognize':
                stop_recognize[0] = True
                soft_asr_blocking[0] = True
                non_speaking[0] = True
                send_every_asr_state_changed(asr_client, soft_asr_blocking[0])

            if soft_asr_blocking[0] and not data_queue.empty():
                logger.debug("clear audio data from queue")
                # clear audio data from queue
                b''.join(data_queue.queue)
                data_queue.queue.clear()
                # notes: start recognize 시에 앞 부분 speech가 잘리는 경우가 있어서 비활성. (주석처리)
                # speech_in_progress[0] = ''

        @staticmethod
        def on_msg_mouth_state_changed(data: str):
            data_obj = json.loads(data)
            logger.debug(f"gRPC event callback: on_msg_mouth_state_changed, data={data_obj},stop={stop_recognize[0]},force={force_start_recognize[0]},once={once_start_recognize[0]}")
            state_value = data_obj['State']
            if state_value == 'Opened':
                mouth_opened[0] = True
            elif state_value == 'Closed':
                mouth_opened[0] = False

            # soft ASR blocking 우선 순위 : stop_recognize > force start recognize > once start recognize > mouth state "closed"
            if not stop_recognize[0] and not force_start_recognize[0] and not once_start_recognize[0]:
                old_value = soft_asr_blocking[0]
                if mouth_opened[0]:
                    soft_asr_blocking[0] = False
                else:
                    soft_asr_blocking[0] = True
                    non_speaking[0] = True
                send_asr_state_changed(asr_client, old_value, soft_asr_blocking[0])

    # start gRPC client listen
    asr_client.start_listen(RpcEventCallback())

    # Cue the user that we're ready to go.
    logger.info("Recorder is ready.")
    logger.info("START\n")

    just_first_time = True
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
                if just_first_time or (phrase_complete and speech_in_progress[0]):
                    just_first_time = False
                    new_speech_started = True
                else:
                    if not speech_in_progress[0]:
                        # case: record finished로 이전 speech 완료된 상태
                        new_speech_started = True

                if new_speech_started:
                    # 새로운 speech 시작
                    phrase_timestamp_string = timestamp_format(phrase_time)
                    logger.debug(f"[{phrase_timestamp_string}] new speech started")
                    send_new_speech_started(asr_client, phrase_timestamp_string)

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
                is_speech_in_progress = bool(speech_in_progress[0])
                logger.debug(f"[{timestamp_string}, {elapsed_time_string}ms, phrase_complete: {phrase_complete}, "
                             f"is_speech_in_progress: {is_speech_in_progress}] {text}")

                if drop_dummy_stt(is_speech_in_progress, text):
                    logger.debug(f"dropped stt={text}")
                else:
                    text = filter_stt(text)

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise, edit the existing one.
                    if phrase_complete and speech_in_progress[0]:
                        # 진행 중이던 speech 완료 처리
                        speech_timestamp_string = timestamp_format(speech_timestamp)
                        logger.debug(f"[{speech_timestamp_string}, phrase_complete] {speech_in_progress[0]}")
                        send_stt(asr_client, speech_timestamp_string, speech_in_progress[0])
                        logger.info(f"[{speech_timestamp_string}] {speech_in_progress[0]}")

                        # 새로운 speech 시작
                        speech_timestamp = phrase_time
                        speech_in_progress[0] = text
                    else:
                        if not speech_in_progress[0]:
                            # record finished로 이전 speech 완료 후 새로운 speech 시작
                            speech_timestamp = phrase_time
                            speech_in_progress[0] = text
                        else:
                            speech_in_progress[0] += ' ' + text
                    logger.debug(f"speech_in_progress={speech_in_progress[0]}")
            else:
                # Infinite loops are bad for processors, must sleep.
                # non-speaking 간주 기준: 1초 동안 listening 하여 0.8초(pause_threshold) 동안 말하지 않거나
                # 더 이상 오디오 입력이 없음.
                time.sleep(1)

                # 1초 후에 non-speaking 상태를 체크한다.
                if non_speaking[0] and speech_in_progress[0]:
                    logger.debug("No speech detected. The phrase is considered complete.")
                    speech_timestamp_string = timestamp_format(speech_timestamp)
                    logger.debug(f"[{speech_timestamp_string}, record_finished] {speech_in_progress[0]}")
                    send_stt(asr_client, speech_timestamp_string, speech_in_progress[0])
                    logger.info(f"[{speech_timestamp_string}] {speech_in_progress[0]}")
                    speech_in_progress[0] = ''

        except KeyboardInterrupt:
            break

    # stop gRPC client listen
    asr_client.stop_listen()

    logger.info("\n\nEND")


def check_microphone_working(mic_name: str) -> bool:
    pyaudio_module = sr.Microphone.get_pyaudio()
    audio = pyaudio_module.PyAudio()
    is_working = False
    try:
        for device_index in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(device_index)
            device_name = device_info.get("name")

            if device_name == mic_name:
                assert (isinstance(device_info.get("defaultSampleRate"), (float, int)) and device_info["defaultSampleRate"] > 0), "Invalid device info returned from PyAudio: {}".format(device_info)
                try:
                    # read audio
                    pyaudio_stream = audio.open(
                        input_device_index=device_index, channels=1, format=pyaudio_module.paInt16,
                        rate=int(device_info["defaultSampleRate"]), input=True
                    )
                    try:
                        buffer = pyaudio_stream.read(1024)
                        if not pyaudio_stream.is_stopped():
                            pyaudio_stream.stop_stream()
                    finally:
                        pyaudio_stream.close()
                except Exception:
                    continue

                # compute RMS of debiased audio
                energy = -audioop.rms(buffer, 2)
                energy_bytes = bytes([energy & 0xFF, (energy >> 8) & 0xFF])
                debiased_energy = audioop.rms(audioop.add(buffer, energy_bytes * (len(buffer) // 2), 2), 2)
                logger.info(f"Microphone name : \"{device_name}\", debiased_energy={debiased_energy}")

                if debiased_energy > 0:  # probably actually audio
                    is_working = True
    finally:
        audio.terminate()
    return is_working


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
        logger.debug("threaded_listen, enter")
        with source as s:
            logger.debug(f"threaded_listen, running={running[0]}")
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = recognizer.listen(s, 1, phrase_time_limit)
                    logger.debug(f"threaded_listen, audio={True if audio is not None else False}")
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


def timestamp_format(dt: datetime):
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def elapsed_time_format(start, end):
    return int(round((end - start) * 1000))


def send_new_speech_started(client, timestamp_string):
    # send a new speech starting point
    data = '{ "Timestamp":"%s" }' % timestamp_string
    response = client.send_message('MSG_NEW_SPEECH_STARTED', data)
    if response:
        logger.debug(f"ASR Client received: status={response.status},message=[{response.message}]")


def send_stt(client, timestamp_string, stt):
    # send stt
    data = '{ "Timestamp":"%s", "Stt":"%s" }' % (timestamp_string, stt)
    response = client.send_message('MSG_ASR', data)
    if response:
        logger.debug(f"ASR Client received: status={response.status},message=[{response.message}]")


def drop_dummy_stt(in_progress, stt):
    return True if not in_progress and stt in dummy_stt else False


def filter_stt(stt):
    m = wuw_pattern.match(stt)
    if m:
        logger.debug(f"WUW match found={m.group()}")
        return stt.replace(m.group(), '').lstrip()
    else:
        return stt


def send_asr_state_changed(client, old_state, new_state):
    logger.debug(f"soft_asr_blocking status : ({old_state} -> {new_state})")
    if old_state and not new_state:
        # case: Soft ASR Blocking true -> false
        asr_state_value = 'AsrProcessing'
    elif not old_state and new_state:
        # case: Soft ASR Blocking false -> true
        asr_state_value = 'SoftAsrBlocking'
    else:
        # case: not changed
        return

    # send ASR state changed
    data = '{ "State":"%s" }' % asr_state_value
    response = client.send_message('MSG_ASR_STATE_CHANGED', data)
    if response:
        logger.debug(f"ASR Client received: status={response.status},message=[{response.message}]")


def send_every_asr_state_changed(client, new_state):
    logger.debug(f"soft_asr_blocking status : {new_state}")
    # notes: soft_asr_blocking status 변경 여부와 상관없이 new state 값을 매번 전송한다.
    if not new_state:
        # case: Soft ASR Blocking: false
        asr_state_value = 'AsrProcessing'
    else:
        # case: Soft ASR Blocking: true
        asr_state_value = 'SoftAsrBlocking'

    # send ASR state changed
    data = '{ "State":"%s" }' % asr_state_value
    response = client.send_message('MSG_ASR_STATE_CHANGED', data)
    if response:
        logger.debug(f"ASR Client received: status={response.status},message=[{response.message}]")


if __name__ == "__main__":
    main()
