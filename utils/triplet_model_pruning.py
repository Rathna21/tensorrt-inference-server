""" Prune triplet model for inference. """
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import logging as 
log
import argparse
import sys

from keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)
from keras import backend as K
from keras.models import (
    Model,
    load_model, 
    save_model
)
from keras.layers import (
    Dense,
    Dropout,
    Input,
    Lambda,
    GlobalAveragePooling2D,
    concatenate,
)
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def lossless_triplet_loss(y_true, y_pred):
    """
    Implementation of Lossless triplet loss function.
    
    Arguments: 
    y_true -- true labels, required when you define a custom loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the embedding for the anchor data
            positive -- the embedding for the positive data (similar to anchor)
            negative -- the embedding for the negative data (different from anchor)
    
    Returns:
    loss -- real number, value of the loss
    """
    # define constants
    max_dist = K.constant(2 * 2)
    epsilon = K.epsilon()
    beta = max_dist
    zero = K.constant(0.0)

    # get the prediction vector
    query, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # compute distance
    pos_distance = K.sum(K.square(query - positive), axis=1)
    neg_distance = K.sum(K.square(query - negative), axis=1)

    # non linear values
    # -ln(-x/N+1)
    pos_dist = -K.log(-((pos_distance) / beta) + 1 + epsilon)
    neg_dist = -K.log(-((max_dist - neg_distance) / beta) + 1 + epsilon)

    # compute loss
    partial_loss = pos_dist + neg_dist
    loss = K.mean(K.maximum(partial_loss, zero), axis=0)

    return loss

def base_network(input_shape=(299, 299, 3), weights="imagenet"):
    """
    Define base network for triplet network
    
    Paramters:
    input_shape(tuple) : Shape of the input image as tuple.
    weights(str) : Pretrained weights to be loaded by the keras.
    
    Returns:
    model(keras.models) : Keras model object.
    """
    base = VGG16(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg"
    )

    # frozen layers
    for layer in base.layers[:10]:
        layer.trainable = False

    # intermediate layers
    layer_names = ["block1_pool", "block2_pool", "block3_pool", "block4_pool"]
    intermediate_layer_outputs = get_layers_output_by_name(base, layer_names)
    convnet_output = base.output
    for layer_name, output in intermediate_layer_outputs.items():
        output = GlobalAveragePooling2D()(output)
        convnet_output = concatenate([convnet_output, output])

    # top layers
    convnet_output = Dense(2048, activation="relu")(convnet_output)
    convnet_output = Dropout(0.6)(convnet_output)
    convnet_output = Dense(2048, activation="relu")(convnet_output)
    convnet_output = Lambda(lambda x: K.l2_normalize(x, axis=1))(convnet_output)

    model = Model(inputs=base.input, outputs=convnet_output, name="base_network")

    return model

def get_layers_output_by_name(model, layer_names):
    return {v: model.get_layer(v).output for v in layer_names}

def triplet_network(base_model, input_shape=(299, 299, 3)):
    """
    Define only single input for the triplet network.
    
    Parameters:
    base_model(keras.models) : base model(VGG16).
    input_shape(tuple) : Shape of the input image as tuple
    
    Returns:
    model(keras.models) : Keras model object.
    """
    # define input: query
    query = Input(shape=input_shape, name="query_input")

    # extract vector using CNN based model
    q_vec = base_model(query)

    # define the triplet model
    model = Model(
        inputs=query, outputs=q_vec, name="output"
    )

    return model

def get_args():
    """
    Function that gets the command line arguments.
    
    Returns:
    All the obtained command line arguments.
    """
    parser = argparse.ArgumentParser(description='Pruning triplet keras model for model inference.')
    parser.add_argument("--keras_model_path", required=True, type=str, help="Path to the saved keras model.(.h5 file)")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to which the pruned model should be saved.")
    parser.add_argument("--pruned_model_name", default="pruned_model", type=str, help="Name of the pruned model.")

    args = parser.parse_args()
    
    return args.keras_model_path, args.out_dir, args.pruned_model_name

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

def prune():
    """
    This function prunes the triplet model into single input.(Inference mode)
    """
    init_logger()
    keras_model_path, out_dir, pruned_model_name = get_args()
    base_model = base_network(input_shape=(299, 299, 3))
    triplet_model = triplet_network(base_model, input_shape=(299, 299, 3))
    
    try:
        if os.path.isfile(keras_model_path):
            orig_model = load_model(keras_model_path, custom_objects={ "lossless_triplet_loss": lossless_triplet_loss })
            
            orig_weights = orig_model.get_weights()
            triplet_model.set_weights(orig_weights)
            save_model(triplet_model, os.path.join(out_dir, pruned_model_name+'.h5'))
            
        else:
            raise FileNotFoundError("KERAS MODEL FILE NOT FOUND at {}".format(keras_model_path))
            
    except FileNotFoundError as err:
        log.error(err.args)
    
if __name__ == "__main__":
    prune()
