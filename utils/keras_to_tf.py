""" Convert Keras model to tensorflow frozen graph. """
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import logging as log
import argparse
import sys
from keras.models import Model, load_model
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util

# python keras-to-tf.py \
# --keras_model_path path_to_keras_model.h5 \
# --out_dir path_to_ouput_directory \ 
# --frozen_graph_name graph_name

def get_args():
    """
    Function that gets the command line arguments.
    
    Returns:
    All the obtained command line arguments.
    """
    parser = argparse.ArgumentParser(description='Convert keras model (.h5) into tensorflow frozen graph(.pb).')
    parser.add_argument("--keras_model_path", required=True, type=str, help="Path to the saved keras model.(.h5 file)")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory to which the converted frozen graph should be saved.")
    parser.add_argument("--frozen_graph_name", default="frozen_graph", type=str, help="Name of the frozen graph.")

    args = parser.parse_args()
    
    return args.keras_model_path, args.out_dir, args.frozen_graph_name

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

def convert():
    """
    This function converts the keras model(.h5) to tensorflow frozen graph(.pb).
    """
    init_logger()
    keras_model_path, out_dir, frozen_graph_name = get_args()
    
    try:
        if os.path.isfile(keras_model_path):
            keras_model = load_model(keras_model_path)
            
            output_node_names = [node.op.name for node in keras_model.outputs]
            log.info("List of output nodes :: ")
            for node in output_node_names:
                log.info(node)
            sess = K.get_session()
            graph = graph_util.convert_variables_to_constants(
                    sess,
                    sess.graph.as_graph_def(),
                    output_node_names)
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok = True)
                
            tf.io.write_graph(graph, out_dir, frozen_graph_name+".pb", as_text=False)
        else:
            raise FileNotFoundError("KERAS MODEL FILE NOT FOUND at {}".format(keras_model_path))
            
    except FileNotFoundError as err:
        log.error(err.args)

if __name__ == "__main__":
    convert()
