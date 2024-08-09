import grpc

import obigo_genesis_ai_poc_pb2 as pb2
import obigo_genesis_ai_poc_pb2_grpc as pb2_grpc


class AsrClient(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        self.channel = grpc.insecure_channel('{}:{}'.format(self.host, self.server_port))
        self.stub = pb2_grpc.AIServiceStub(self.channel)

    def send_message(self, method, data):
        try:
            message = pb2.RequestMessage(method=method, data=data)
            return self.stub.Message(message)
        except grpc.RpcError as e:
            print(f"gRPC error code={e.code()}, details={e.details()}")


if __name__ == '__main__':
    client = AsrClient()
    print("Will try to request a TEST message to the AI Service Server...")

    response = client.send_message('MSG_ASR', 'stt - my test')
    if response:
        print(f"ASR Client received: status={response.status},message=[{response.message}]")
