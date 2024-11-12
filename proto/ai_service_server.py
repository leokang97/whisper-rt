import logging
import time
from concurrent import futures
from typing import Iterable

import grpc
from google.protobuf.json_format import MessageToJson

import obigo_genesis_ai_poc_pb2 as pb2
import obigo_genesis_ai_poc_pb2_grpc as pb2_grpc

logger = logging.getLogger('ai_service_server')
logger.setLevel(logging.DEBUG)

# console logger
formatter = logging.Formatter('%(asctime)s %(module)s:%(levelname)s:%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_simulator_event(mic_on: bool) -> pb2.EventMessage():
    event = pb2.EventMessage()
    event.method = 'MSG_CONTROL'
    event_name = 'startRecognize' if mic_on else 'stopRecognize'
    event.data = '{ "event":"%s" }' % event_name
    logger.info(f"Created a SIMULATOR event={MessageToJson(event)}")
    return event


def create_webapp_event(once_seconds: int) -> pb2.EventMessage():
    event = pb2.EventMessage()
    event.method = 'MSG_CONTROL'
    event.data = '{ "event":"startRecognize", "once":%d }' % once_seconds
    logger.info(f"Created a SIMULATOR event={MessageToJson(event)}")
    return event


def create_camera_event(mouth_opened: bool) -> pb2.EventMessage():
    event = pb2.EventMessage()
    event.method = 'MSG_MOUTH_STATE_CHANGED'
    state_value = 'Opened' if mouth_opened else 'Closed'
    event.data = '{ "State":"%s" }' % state_value
    logger.info(f"Created a CAMEAR event={MessageToJson(event)}")
    return event


class AiServiceServer(pb2_grpc.AIServiceServicer):

    def __init__(self, *args, **kwargs):
        pass

    def Message(self, request, context):
        logger.info("Received a RequestMessage")
        response = {'status': True, 'message': f'method: "{request.method}", data: "{request.data}"'}
        return pb2.ResponseMessage(**response)

    def Event(self, request, context) -> Iterable[pb2.EventMessage]:
        logger.info("Received a subscription request")
        # testcase: normal
        # Simulate mic off
        time.sleep(3)
        logger.info("Mic Off")
        yield create_simulator_event(False)
        # Simulate mic on
        time.sleep(3)
        logger.info("Mic On")
        yield create_simulator_event(True)
        # Simulate driver mouth closed
        time.sleep(3)
        logger.info("Driver Mouth Closed")
        yield create_camera_event(False)
        # Simulate driver mouth opened
        time.sleep(3)
        logger.info("Driver Mouth Opened")
        yield create_camera_event(True)
        # Simulate mic off
        time.sleep(3)
        logger.info("Mic Off")
        yield create_simulator_event(False)

        # testcase: start recognize > once [n] seconds
        # time.sleep(3)
        # logger.info("start recognize > once 10 seconds")
        # yield create_webapp_event(10)
        # time.sleep(3)
        # logger.info("start recognize > once 5 seconds")
        # yield create_webapp_event(5)
        # time.sleep(1)
        # logger.info("Driver Mouth Closed")
        # yield create_camera_event(False)
        # time.sleep(1)
        # logger.info("Driver Mouth Opened")
        # yield create_camera_event(True)
        # time.sleep(1)
        # logger.info("Driver Mouth Closed")
        # yield create_camera_event(False)
        # time.sleep(3)
        # logger.info("Driver Mouth Closed")
        # yield create_camera_event(False)


def serve(address: str):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_AIServiceServicer_to_server(AiServiceServer(), server)
    server.add_insecure_port(address)
    server.start()
    logger.info(f"AI Service Server started, serving at {address}")
    server.wait_for_termination()


if __name__ == '__main__':
    port = '50051'
    serve('[::]:' + port)
