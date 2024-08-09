from concurrent import futures

import grpc

import obigo_genesis_ai_poc_pb2 as pb2
import obigo_genesis_ai_poc_pb2_grpc as pb2_grpc


class AiServiceServer(pb2_grpc.AIServiceServicer):

    def __init__(self, *args, **kwargs):
        pass

    def Message(self, request, context):
        response = {'status': True, 'message': f'method: "{request.method}", data: "{request.data}"'}
        return pb2.ResponseMessage(**response)


def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_AIServiceServicer_to_server(AiServiceServer(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("AI Service Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
