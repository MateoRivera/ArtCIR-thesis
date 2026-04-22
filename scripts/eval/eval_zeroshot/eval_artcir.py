from artcir_thesis.milvus import MilvusWrapper
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Libraries for the model
import torch
import numpy as np
from artcir_thesis.dataset import ArtCIRDataset
import torch.nn.functional as F 


def build_dataset(args, type_: str):
    return ArtCIRDataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type_=type_,
        split=args.split
    )

def recall(args):
    # Load the dataset and the model
    query_dataset = build_dataset(args, type_='query')
    
    ks = [1, 5, 10, 25, 50]
    recall_at = {k: [] for k in ks}
    map_at = {k: [] for k in ks}
    
    with MilvusWrapper(args.db_path) as db:
        query_collection = 'query_embeddings'
        image_collection = 'image_embeddings'

        for triplet in tqdm(query_dataset.annotations):
            # Retrieve the reference image's embedding
            query_embedding = db.client.query(
                collection_name=query_collection,
                filter=f"artefact_id == '{triplet['reference_qid']}' AND split == '{args.split}'",
                output_fields=["vector"],
            )[0]['vector']

            # Retrieve the top 50 most similar images
            top50 = db.search(
                collection_name=image_collection,
                query_vector=query_embedding,
                top_k=50,
                output_fields=["artefact_id"]
            )
            top50_qids = [hit['entity']['artefact_id'] for hit in top50[0]]

            # Compare the results with the ground truth
            target_qid = triplet['target_qid']
            
            for k in ks:
                recall_at[k].append(1 if target_qid in top50_qids[:k] else 0)
                map_at[k].append(1 / (1 + top50_qids[:k].index(target_qid)) if target_qid in top50_qids[:k] else 0)


    for k in ks:
        recall_at[k] = np.mean(recall_at[k]) * 100
        map_at[k] = np.mean(map_at[k]) * 100

    print(recall_at)
    print(map_at)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path_prefix', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    # parser.add_argument('--type', choices=['query', 'image'], type=str)
    parser.add_argument('--split', choices=['train', 'val', 'test'], type=str)
    # parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--model_id', choices=["Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B"], type=str)
    parser.add_argument('--db_path', type=Path)
    args = parser.parse_args()    

    recall(args)