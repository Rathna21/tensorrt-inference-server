from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config
import stubs.inference_pb2 as inference_pb2
import stubs.inference_pb2_grpc as inference_pb2_grpc
import utils
import logging as log
from PIL import Image
import io
import grpc
import numpy as np
import tensorflow as tf
import pickle
from annoy import AnnoyIndex
from keras.applications.vgg16 import preprocess_input
import time
import os


class FashionMatchingBottom(inference_pb2_grpc.InferenceAPIServicer):
    """ This is a class for retrieving the similar images from the catalogue to that of the input image. """
    def __init__(self):
        """ Set Parameters corresponding the gRPC end point of the tensorrt inference server. """
        # Within the Docker container => trt-server:8001
        # Without Docker container => localhost:8001
        self._URL = "trt-server:8001"
        self._MODEL_NAME = "fashion_matching_bottom"
        self._BATCH_SIZE = 1
        # -1 indicates that latest version in the model repository is pointed to.
        self._MODEL_VERSION = -1
        self._EMBEDDING_SIZE = 2048
        # size to which the input image has to be resized. This size is the width
        # and height that the model expects.
        self._DESIRED_SIZE = 299
        # K nearest neighbour
        self._K = 10

    def fashion_matching_preprocess(self, img):
        """
        Preprocess the input image.
        
        Arguments:
        img(PIL.Image): Input image.
        
        Returns:
        img(numpy array): Preprocessed 4d numpy array of the input image.
        """
        old_size = [self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
        ratio = float(self._DESIRED_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new("RGB", (self._DESIRED_SIZE, self._DESIRED_SIZE))
        new_img.paste(
            img,
            (
                (self._DESIRED_SIZE - new_size[0]) // 2,
                (self._DESIRED_SIZE - new_size[1]) // 2,
            ),
        )
        img = np.asarray(new_img, np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        return img

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
        request.meta_data.batch_size = self._BATCH_SIZE
        request.model_version = self._MODEL_VERSION
        output_list = ["base_network/lambda_1/l2_normalize"]

        for output in output_list:
            output_message = api_pb2.InferRequestHeader.Output()
            output_message.name = output
            request.meta_data.output.extend([output_message])

        image_data = []
        image_data.append(self.fashion_matching_preprocess(img))
        request.meta_data.input.add(name="query_input", dims=[299, 299, 3])
        input_bytes = image_data[0].tobytes()
        request.raw_input.extend([input_bytes])

        return request

    def fashion_matching_postprocess(self, im_emb):
        """
        Postprocessing the model response.
        
        Arguments:
        im_emb(numpy array) : Embedding generated for the test image.(2048 dimension feature vector).
        
        Returns:
        KNN(list) : list of 10 nearest neighbors of the test image as python list.
        """
        # embeddings = utils.Utils.get_embeddings(self, self._MODEL_NAME)
        search_index = utils.Utils.load_annoy_index(self, self._MODEL_NAME)
        KNN = search_index.get_nns_by_vector(im_emb, self._K)
        log.info(KNN)

        return KNN

    def fashion_matching_request(self, img):
        """
        Sends request to the running model at the tensorrt inference server and returns the response.
        
        Arguments:
        img(PIL.Image) : Input image.
        
        Returns:
        reco_res(list) : 10 nearest neighbours of the test image as python list.
        """
        with grpc.insecure_channel(self._URL) as channel:
            grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            request = self.prepare_request(img)
            response = grpc_stub.Infer(request)
            im_emb = np.frombuffer(response.raw_output[0], np.float32)
            reco_res = self.fashion_matching_postprocess(im_emb)

        return reco_res

    def GetFashionMatchingBottom(self, request, context):
        """
        This function takes request from the request dispatcher and sends back the response to it.
        
        Arguments:
        request : gRPC request object.
        context : gRPC context object.
        
        Returns:
        resp : gRPC response object containing the 10 similar results.
        """
        log.info("Fashion Matching Bottom Request")
        img = utils.Utils.bytes_to_image(self, request.image)
        reco_list = self.fashion_matching_request(img)
        resp = inference_pb2.FashionMatchingBottom()
        resp.reco.extend(reco_list)

        return resp
