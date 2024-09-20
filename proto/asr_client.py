import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import grpc

import obigo_genesis_ai_poc_pb2 as pb2
import obigo_genesis_ai_poc_pb2_grpc as pb2_grpc

logger = logging.getLogger('asr_client')
logger.setLevel(logging.DEBUG)

# console logger
formatter = logging.Formatter('%(asctime)s %(module)s:%(levelname)s:%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

RETRY_COUNT = 10
RETRY_INTERVAL = 10  # seconds


class AsrClient(object):
    def __init__(self):
        self._host = 'localhost'
        self._server_port = 50051
        self._executor = ThreadPoolExecutor()
        self._channel = grpc.insecure_channel('{}:{}'.format(self._host, self._server_port))
        self._stub = pb2_grpc.AIServiceStub(self._channel)
        self._finished = threading.Event()
        self._consumer_future = None

    def send_message(self, method, data):
        try:
            message = pb2.RequestMessage(method=method, data=data)
            return self._stub.Message(message)
        except grpc.RpcError as e:
            logger.warning(f"gRPC Message response error={e}")

    def _response_watcher(self, response_iterator: Iterable[pb2.EventMessage]) -> None:
        try:
            for response in response_iterator:
                if response.method == 'MSG_CONTROL':
                    self._on_msg_control(response.data)
                else:
                    logger.warning(f"invalid method={response.method}")
        except Exception as e:
            logger.warning(f"gRPC Event response error={e}")
            self._finish()
            raise

    def _on_msg_control(self, data: str) -> None:
        logger.info(f"MSG_CONTROL received: data={data}")

    def _subscribe_events(self):
        logger.info("subscribe to events")
        request = pb2.google_dot_protobuf_dot_empty__pb2.Empty()
        response_iterator = self._stub.Event(request)

        # Instead of consuming the response on current thread, spawn a consumption thread.
        self._consumer_future = self._executor.submit(
            self._response_watcher, response_iterator
        )

    def _wait_events(self) -> bool:
        """
        :return: True is Event response watcher에서 exception이 발생하여 retry가 필요, otherwise False
        """
        logger.info("Waiting to receive event...")
        need_to_retry = False
        self._finished.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            exception = self._consumer_future.exception()
            if exception:
                need_to_retry = True
                logger.info(
                    f"Event response watcher failed, error code={exception.code()}, details={exception.details()}")
            else:
                need_to_retry = False
                self._consumer_future.result()
                logger.info("Event response watcher done")

        logger.info(f"Event response watcher finished, need to retry={need_to_retry}")
        self._finished.clear()
        return need_to_retry

    def _finish(self):
        self._finished.set()

    def _client_listener(self) -> None:
        retry_subscribe = False
        for retry in range(0, RETRY_COUNT + 1):
            if retry > 0:
                logger.info(f"retry={retry}")

            self._subscribe_events()
            retry_subscribe = self._wait_events()
            if not retry_subscribe:
                break

            if retry < RETRY_COUNT:
                time.sleep(RETRY_INTERVAL)

        if retry_subscribe:
            logger.info(f"Retry {RETRY_COUNT} times, Event subscription failed")

        logger.info("client listener finished")

    def start_listen(self) -> None:
        logger.info("start listen")
        self._executor.submit(
            self._client_listener
        )

    def stop_listen(self) -> None:
        logger.info("stop listen")
        self._finish()


if __name__ == '__main__':
    client = AsrClient()
    logger.info("Will try to request a TEST message to the AI Service Server...")

    response_message = client.send_message('MSG_ASR', 'stt - my test')
    if response_message:
        logger.info(f"ASR Client received: status={response_message.status},message=[{response_message.message}]")

    client.start_listen()

    time.sleep(RETRY_COUNT * RETRY_INTERVAL + 10)
    client.stop_listen()
