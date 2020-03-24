import sys

sys.path.append("stubs/")
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config
import inference_pb2
import inference_pb2_grpc
import utils
import logging as log
from PIL import Image
import io
import grpc
import numpy as np
import tensorflow as tf


class Classification(inference_pb2_grpc.InferenceAPIServicer):
    """ This class returns the classification result of the input image. """
    def __init__(self):
        """ Set Parameters corresponding the gRPC end point of the tensorrt inference server. """
        self._IMAGE_WIDTH = 0
        self._IMAGE_HEIGHT = 0
        # Within the Docker container => trt-server:8001
        # Without Docker container => localhost:8001
        self._URL = "trt-server:8001"
        self._MODEL_NAME = "vgg16"
        self._BATCH_SIZE = 1
        # -1 points the model to the latest version in the model repository.
        self._MODEL_VERSION = -1

    def prepare_request(self, img):
        """
        Prepare gRPC request for model inference with the tensorrt inference server.
        
        Arguments:
        img(PIL.Image): Input image.
        
        Returns:
        request(gRPC Request object): Request object with all information about the request.
        """
        request = grpc_service_pb2.InferRequest()
        request.model_name = self._MODEL_NAME
        request.model_version = self._MODEL_VERSION
        request.meta_data.batch_size = self._BATCH_SIZE
        output_message = api_pb2.InferRequestHeader.Output()
        output_message.name = "vgg0_dense2_fwd"
        request.meta_data.output.extend([output_message])
        image_data = []
        image_data.append(self.vgg16_preprocess(img))
        request.meta_data.input.add(name="data")
        input_bytes = image_data[0].tobytes()
        request.raw_input.extend([input_bytes])

        return request

    def classification_request(self, img):
        """
        Sends request to the running model at the tensorrt inference server and returns the response.
        
        Arguments:
        img(PIL.Image) : Input image.
        
        Returns:
        index(int) : Index of the predicted class.
        """
        with grpc.insecure_channel(self._URL) as channel:
            grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            request = self.prepare_request(img)
            response = grpc_stub.Infer(request)
            result = response.raw_output[0]
            res_arr = np.frombuffer(result, dtype=np.float32)
            res_arr = np.reshape(res_arr, (1000,))
            index = np.argmax(utils.Utils.softmax(self, res_arr))

            return index

    def vgg16_preprocess(self, img):
        """
        Preprocess the input image.
        
        Arguments:
        img(PIL.Image): Input image.
        
        Returns:
        img(numpy array): Preprocessed 3d numpy array of the input image.
        """
        img_rgb = img.convert("RGB")
        # VGG16 model expects input size 224 x 224.
        img = img_rgb.resize((224, 224))
        self._IMAGE_WIDTH, self._IMAGE_HEIGHT = img.size
        img_arr = np.array(img.getdata()).reshape(
            (self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 3)
        )
        # VGG16 model normalizes the input between -1 and +1.
        img_arr = img_arr / 127.5 - 1
        # ONNX expects NCHW (channel first input).
        img_arr = np.transpose(img_arr, (2, 0, 1))
        # VGG16 model expects input data type as float32.
        img_arr = img_arr.astype(np.float32)

        return img_arr

    def GetClassification(self, request, context):
        """
        This function takes request from the request dispatcher and sends back the response to it.
        
        Arguments:
        request : gRPC request object.
        context : gRPC context object.
        
        Returns:
        resp : Taxonomy of the classification result.
        """
        taxonomy = inference_pb2.ImageClassification()
        img = utils.Utils.bytes_to_image(self, request.image)
        response = self.classification_request(img)
        taxonomy.taxonomy = "class " + str(response)

        return taxonomy

        # Status Request
        # response_server_status = utils.Utils.status_request(self)
        # print(response_server_status)
