import numpy as np
from PIL import Image
import grpc
import io
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
import tensorrtserver.api.model_config_pb2 as model_config
import inference_pb2
import inference_pb2_grpc
import logging as log
import os
from annoy import AnnoyIndex
import pickle


class Utils:
    @staticmethod
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum(axis=0)

    @staticmethod
    def status_request(self):
        with grpc.insecure_channel(self._URL) as channel:
            grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            request = grpc_service_pb2.StatusRequest(model_name=self._MODEL_NAME)
            response = grpc_stub.Status(request)
            return response

    @staticmethod
    def bytes_to_image(self, img):
        img = Image.open(io.BytesIO(img))
        self._IMAGE_WIDTH = img.size[0]
        self._IMAGE_HEIGHT = img.size[1]
        log.info(self._IMAGE_WIDTH)
        log.info(self._IMAGE_HEIGHT)

        return img

    @staticmethod
    def image_to_bytes(self, img):
        img = img.convert("RGB")
        io_buf = io.BytesIO()
        img.save(io_buf, format="JPEG")
        byte_im = io_buf.getvalue()

        return byte_im

    @staticmethod
    def create_annoy_index(self, model_name):

        if model_name not in self._MODEL_LIST:
            search_index = AnnoyIndex(self._EMBEDDING_SIZE, metric="euclidean")

            for i, emdedding in embeddings.items():
                search_index.add_item(i, emdedding)
            search_index.build(50)
            search_index.save(
                os.path.join(
                    os.getcwd(),
                    "opt/inf-ser/annoy_index/{}_annoy_index.ann".format(model_name),
                )
            )

            self._MODEL_LIST.append(model_name)
            log.info("{} appended to the model list.".format(model_name))

        return None

    @staticmethod
    def load_annoy_index(self, model_name):
        search_index = AnnoyIndex(self._EMBEDDING_SIZE, metric="euclidean")
        search_index.load(
            os.path.join(
                os.getcwd(),
                "opt/inf-ser/annoy_index/{}_annoy_index.ann".format(model_name),
            )
        )

        return search_index

    @staticmethod
    def get_embeddings(self, model_name):
        with open(
            os.path.join(
                os.getcwd(), "opt/inf-ser/catalog_embedding/{}_embed".format(model_name)
            ),
            "rb",
        ) as f:
            embeddings = pickle.load(f)

        return embeddings
