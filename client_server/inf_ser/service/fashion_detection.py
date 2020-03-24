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
import fashion_matching_bottom
import fashion_matching_top
import fashion_matching_full
import utils


class FashionDetection(inference_pb2_grpc.InferenceAPIServicer):
    """ This is a class that performs detection of fashion objects in the given image. """
    def __init__(self):
        """
        Set Parameters corresponding the gRPC end point of the tensorrt inference server.
        """
        self._LABELS_DICT = {
            1.0: "coat",
            2.0: "dress",
            3.0: "gown",
            4.0: "jacket",
            5.0: "sweater",
            6.0: "shirt",
            7.0: "t-shirt",
            8.0: "jeans",
            9.0: "pants",
            10.0: "skirt",
            11.0: "suit",
            12.0: "sweatshirt",
            13.0: "footwear",
            14.0: "bag",
            15.0: "glasses",
            16.0: "headwear",
            17.0: "jewelry",
        }
        self._MAX_AREA = 1.0
        self._MIN_TEMP_SCORE = 0.3
        self._IMAGE_WIDTH = 0
        self._IMAGE_HEIGHT = 0
        # Within the Docker container => trt-server:8001
        # Without Docker container => localhost:8001
        self._URL = "trt-server:8001"
        self._MODEL_NAME = "fashion-detection"
        self._BATCH_SIZE = 1
        # -1 indicates that latest version in the model repository is pointed to.
        self._MODEL_VERSION = -1

    def check_size_and_labels(self, boxes, scores, classes):
        """
        This method checks the size of the bounding boxes returned and element the incorrect boxes based on the area.
        
        Arguments:
        boxes(numpy_array) : Predicted bounding boxes.
        scores(numpy_array) : Predicted confidence scores for each boxes.
        classes(numpy_array) : Predicted classes.
        
        Returns:
        boxes_new(numpy_array) : Valid bounding boxes
        scores_new(numpy_array) : Scores for the valid boxes.
        classes_new(numpy_array) : Classes for the valid boxes.
        """
        boxes_new, scores_new, classes_new = [], [], []

        for box, score, cl in zip(boxes[0], scores[0], classes[0]):
            area = (box[3] - box[1]) * (box[2] - box[0])

            if (
                area < self._MAX_AREA
                and score > self._MIN_TEMP_SCORE
                and cl in self._LABELS_DICT.keys()
            ):
                boxes_new.append(list(box))
                scores_new.append(score)
                classes_new.append(cl)

        return (
            np.expand_dims(np.array(boxes_new), axis=0),
            np.expand_dims(np.array(scores_new), axis=0),
            np.expand_dims(np.array(classes_new), axis=0),
        )

    def fashion_detection_postprocess(self, boxes, scores, classes, img):
        """
        This method does postprocessing of the fashion detection model.
        
        Arguments:
        boxes(numpy_array) : Predicted bounding boxes.
        scores(numpy_array) : Predicted confidence scores for each boxes.
        classes(numpy_array) : Predicted classes.
        img(PIL.Image) : Input Image.
        
        Returns:
        result_dict(dict) : Returns dictionary with the bounding boxes, classes and confidence scores.
        """
        boxes, scores, classes = self.check_size_and_labels(boxes, scores, classes)
        log.info("no. of boxes detected :: {}".format(boxes.size))
        result_dict = {
            "class_list": [],
            "score_list": [],
            "ymin_list": [],
            "xmin_list": [],
            "ymax_list": [],
            "xmax_list": [],
        }

        if boxes.size == 0:
            return result_dict

        with tf.compat.v1.Session() as sess:
            selected_indices = tf.image.non_max_suppression(
                boxes[0], scores[0], 100, 0.3
            )
            selected_boxes = tf.gather(boxes[0], selected_indices)
            selected_boxes = sess.run(selected_boxes)

        for box, score, cl in zip(selected_boxes, scores[0], classes[0]):
            ymin = int(box[0] * self._IMAGE_HEIGHT)
            xmin = int(box[1] * self._IMAGE_WIDTH)
            ymax = int(box[2] * self._IMAGE_HEIGHT)
            xmax = int(box[3] * self._IMAGE_WIDTH)

            result_dict["class_list"].append(self._LABELS_DICT[cl])
            result_dict["score_list"].append(score)
            result_dict["ymin_list"].append(ymin)
            result_dict["xmin_list"].append(xmin)
            result_dict["ymax_list"].append(ymax)
            result_dict["xmax_list"].append(xmax)

        return result_dict

    def fashion_detection_preprocess(self, img):
        """
        Preprocess the input image.
        
        Arguments:
        img(PIL.Image): Input image.
        
        Returns:
        img(numpy array): Preprocessed numpy array of the input image.
        """
        img_rgb = img.convert("RGB")

        return (
            np.array(img_rgb.getdata())
            .reshape((self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 3))
            .astype(np.uint8)
        )

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
        output_list = ["detection_boxes", "detection_scores", "detection_classes"]

        for output in output_list:
            output_message = api_pb2.InferRequestHeader.Output()
            output_message.name = output
            request.meta_data.output.extend([output_message])

        image_data = []
        image_data.append(self.fashion_detection_preprocess(img))
        request.meta_data.input.add(
            name="image_tensor", dims=[self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 3]
        )
        input_bytes = image_data[0].tobytes()
        request.raw_input.extend([input_bytes])

        return request

    def fashion_detection_request(self, img):
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
            detection_boxes = np.zeros((self._BATCH_SIZE, 300, 4), np.float32)
            detection_classes = np.zeros((self._BATCH_SIZE, 300), np.float32)
            detection_scores = np.zeros((self._BATCH_SIZE, 300), np.float32)

            if len(response.raw_output) > 0:
                detection_boxes = np.reshape(
                    np.frombuffer(response.raw_output[0], np.float32),
                    (self._BATCH_SIZE, 300, 4),
                )
                detection_classes = np.reshape(
                    np.frombuffer(response.raw_output[1], np.float32),
                    (self._BATCH_SIZE, 300),
                )
                detection_scores = np.reshape(
                    np.frombuffer(response.raw_output[2], np.float32),
                    (self._BATCH_SIZE, 300),
                )

                res = self.fashion_detection_postprocess(
                    detection_boxes, detection_scores, detection_classes, img
                )

            else:
                res = self.fashion_detection_postprocess(
                    detection_boxes, detection_scores, detection_classes, img
                )

            return res

    def GetFashionDetection(self, request, context):
        """
        This function takes request from the request dispatcher and sends back the response to it.
        
        Arguments:
        request : gRPC request object.
        context : gRPC context object.
        
        Returns:
        resp : gRPC response object containing detection results.
        """
        log.info("Fashion Detection Request")
        detection = inference_pb2.FashionDetection()
        img = utils.Utils.bytes_to_image(self, request.image)
        res = self.fashion_detection_request(img)

        for i in range(len(res["class_list"])):
            detection.detections.add(
                taxonomy=res["class_list"][i],
                confidence_score=res["score_list"][i],
                y_min=res["ymin_list"][i],
                x_min=res["xmin_list"][i],
                y_max=res["ymax_list"][i],
                x_max=res["xmax_list"][i],
                type="box",
            )

        detection.error = ""

        return detection
