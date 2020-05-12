"""The Python implementation of the GRPC Inference server."""
import sys

sys.path.append("/opt/inf-ser/stubs/")
from concurrent import futures
import time
import grpc
import inference_pb2
import inference_pb2_grpc
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from functools import partial
import os
import logging as log
import object_classification
import fashion_matching_mnist


class RequestDespatcher(inference_pb2_grpc.InferenceAPIServicer):
    """
    This is the class for dispatching the request to the corresponding services.
    """
    def GetFashionMatchingMNIST(self, request, context):
        log.info("Fashion Matching MNIST Request")

        return fashion_matching_mnist.FashionMatchingMNIST().GetFashionMatchingMNIST(
            request, context
        )

    def GetClassification(self, request, context):
        log.info("Object Classification Request")

        return object_classification.Classification().GetClassification(
            request, context
        )


def serve():
    """
    Function for running the server for 24 hours.
    """
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceAPIServicer_to_server(RequestDespatcher(), server)
    server.add_insecure_port("[::]:50053")
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


def init_logger():
    """
    Function that initializes the logger.
    """
    log.basicConfig()

    root_log = log.getLogger()
    root_log.setLevel(log.INFO)
    fmt = log.Formatter("%(levelname)s:%(message)s")
    stdout = log.StreamHandler(stream=sys.stdout)
    stdout.setFormatter(fmt)
    root_log.addHandler(stdout)


if __name__ == "__main__":
    init_logger()
    serve()
