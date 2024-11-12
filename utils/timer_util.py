import threading
import time


class MyTimer:
    def __init__(self):
        self._seconds = 0
        self._finish_callback = None
        self._running = False
        self._timer_thread = None

    def _timer(self):
        # timer thread를 중지하기 위해서 sleep (100ms) 하여 즉시 중단되도록 함
        loop_count = 0
        finish_count = self._seconds * 10
        while self._running and loop_count < finish_count:
            loop_count += 1
            time.sleep(0.1)

        # timer가 완료된 경우에만 finish callback을 호출함
        if self._finish_callback is not None and loop_count == finish_count:
            self._finish_callback()
        self._running = False

    def _stop(self):
        self._running = False
        self._timer_thread.join()

    def start_timer(self, seconds, callback=None):
        if self._running:
            self._stop()

        self._seconds = seconds
        self._finish_callback = callback
        self._running = True

        self._timer_thread = threading.Thread(target=self._timer)
        self._timer_thread.start()
