""" Create Annoy Index for the Embedding. """
import logging as log
import argparse
import sys
import os
from annoy import AnnoyIndex
import pickle
import time

# python create_annoy_index.py \
# --name fashion_matching_model \
# --embed_path directory_embedding_path/embedding_file \
# --embed_size 2048 \
# --out_dir output_directory_to_store_the_annoy_index \
# --n_trees 50

def create_annoy_index(name, embed_path, embed_size, out_dir, metric, n_trees):
    
    """
    This function creates annoy index for the given embedding.

    Parameters:
    name(str) : Model name.
    embed_path(pkl) : Path to the embedding (pickle) file.
    embed-size(int) : Dimension of the embeddings.
    out_dir(str) : Path to the output directory of the generated annoy file.
    n_trees(int) : Number of trees to be built for the annoy index.

    """
    
    search_index = AnnoyIndex(embed_size, metric=metric)
        
    try:
        if os.path.isfile(embed_path):
            with open(embed_path,'rb') as f:
                embeddings = pickle.load(f)
                
            log.info("Building Annoy index...")
            start_time = time.process_time()
            for i, emdedding in embeddings.items():
                search_index.add_item(i, emdedding)   
            search_index.build(n_trees)
            log.info("{} seconds taken for building the annoy index with {} trees.".format(str(round((time.process_time() - start_time), 2)), str(n_trees)))
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok = True)
                
            output_path = os.path.join(out_dir, "{}_annoy_index.ann".format(name))
            search_index.save(output_path)
            log.info("Annoy index created in the path {}".format(output_path))
                
        else:
            raise FileNotFoundError("EMBED FILE NOT FOUND at {}".format(embed_path))
        
    except FileNotFoundError as err:
        log.error(err.args)
    
def get_args():
    """
    Function that gets the command line arguments.
    
    Returns:
    All the obtained command line arguments.
    """
    parser = argparse.ArgumentParser(description='Create Annoy Index for the embedding.')
    parser.add_argument("--name", default="annoy_index", type=str, help="Embedding model name for which annoy index is created.")
    parser.add_argument("--embed_path", required=True, type=str, help="Path to the pickled embedding dictionary.")
    parser.add_argument("--embed_size", required=True, type=int, help="Size of the image embedding.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory path to the annoy index.")
    parser.add_argument("--metric", choices=["angular", "euclidean", "manhattan", "hamming", "dot"], 
                        default="euclidean", help="Metric for calculating the distance of the annoy index.")
    parser.add_argument("--n_trees", default=50, type=int, help="No. of trees for building the annoy index.")
    args = parser.parse_args()
    
    return args.name, args.embed_path, args.embed_size, args.out_dir, args.metric, args.n_trees

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
    name, embed_path, embed_size, out_dir, metric, n_trees = get_args()
    create_annoy_index(name, embed_path, embed_size, out_dir, metric, n_trees)
