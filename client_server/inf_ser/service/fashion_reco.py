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
import fashion_detection
import fashion_matching_bottom
import fashion_matching_top
import fashion_matching_full
import utils


class FashionReco(inference_pb2_grpc.InferenceAPIServicer):
    def __init__(self):
        self._LABEL_MAP = {
            "top": ["coat", "jacket", "sweater", "shirt", "t-shirt", "sweatshirt"],
            "bottom": ["jeans", "pants", "skirt"],
            "full": ["dress", "gown", "suit"],
        }

    def get_crop(self, detection, img):
        im_crop = img.crop(
            (detection.x_min, detection.y_min, detection.x_max, detection.y_max)
        )

        return im_crop

    def GetFashionRecommendations(self, request, context):
        log.info("Fashion Detection Request")
        reco = inference_pb2.FashionRecommendations()
        # Get the objects detected from the fashion detection model.
        detections = fashion_detection.FashionDetection().GetFashionDetection(
            request, context
        )
        img = utils.Utils.bytes_to_image(self, request.image)
        for detection in detections.detections:
            # Get the crop of the detected object.
            img_crop = self.get_crop(detection, img)
            im_bytes = utils.Utils.image_to_bytes(self, img_crop)
            request.image = im_bytes

            if detection.taxonomy in self._LABEL_MAP["top"]:
                reco_resp = fashion_matching_top.FashionMatchingTop().GetFashionMatchingTop(
                    request, context
                )
            elif detection.taxonomy in self._LABEL_MAP["bottom"]:
                reco_resp = fashion_matching_bottom.FashionMatchingBottom().GetFashionMatchingBottom(
                    request, context
                )
            elif detection.taxonomy in self._LABEL_MAP["full"]:
                reco_resp = fashion_matching_full.FashionMatchingFull().GetFashionMatchingFull(
                    request, context
                )
            else:
                continue

            reco_list = reco_resp.reco
            reco.fashionObjects.add(taxonomy=detection.taxonomy, reco=reco_list)
            del reco_list, reco_resp.reco[:]

        log.info(reco.fashionObjects)
        return reco
