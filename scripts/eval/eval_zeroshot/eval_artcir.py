from artcir_thesis.milvus import MilvusWrapper
from pathlib import Path
import argparse
from tqdm import tqdm
import json

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

    retrieval_results = {'top50_predictions_by_reference_qid': {}}
    
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

            # Store the top 50 per each triplet
            retrieval_results['top50_predictions_by_reference_qid'][triplet['reference_qid']] = \
                top50_qids

            # Compare the results with the ground truth
            target_qid = triplet['target_qid']
            
            for k in ks:
                recall_at[k].append(1 if target_qid in top50_qids[:k] else 0)
                map_at[k].append(1 / (1 + top50_qids[:k].index(target_qid)) if target_qid in top50_qids[:k] else 0)


    for k in ks:
        recall_at[k] = np.mean(recall_at[k]) * 100
        map_at[k] = np.mean(map_at[k]) * 100

    retrieval_results['summary_metrics'] = {
        'recall_at': recall_at,
        'map_at': map_at
    }

    output_path = args.retrieval_results_path_prefix / f'retrieval_results_{args.split}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as retrieval_results_file:
        # Use indent=4 for human-readable output, remove for a smaller file size
        json.dump(retrieval_results, retrieval_results_file, ensure_ascii=False, indent=4)

    print('recall@:', recall_at)
    print('map@:', map_at)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path_prefix', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--split', choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--db_path', type=Path)
    parser.add_argument('--retrieval_results_path_prefix', type=Path)
    args = parser.parse_args()    

    recall(args)