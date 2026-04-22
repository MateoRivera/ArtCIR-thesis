from artcir_thesis.milvus import MilvusWrapper
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Libraries for the model
import torch
from models.qwen3_vl_embedding import Qwen3VLEmbedder
from artcir_thesis.dataset import ArtCIRDataset
from torch.utils.data import DataLoader 

def qwen_message_collate_fn(batch):
    # batch is a list of dataset items: [(message_dict, idx), ...]
    messages, ids = zip(*batch) # (message_dict1, ...), (idx1, ...)
    return list(messages), list(ids)

def build_dataset(args):
    return ArtCIRDataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type_=args.type,
        split=args.split
    )
def build_dataloader(args, dataset):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qwen_message_collate_fn)

def embed(args):
    # Load the dataset and the model
    dataset = build_dataset(args)
    dataloader = build_dataloader(args, dataset)
    model = Qwen3VLEmbedder(model_name_or_path=args.model_id,
                            torch_dtype=torch.float16
            )
    print("Model device:", model.model.device, flush=True)
    
    # Check if device is GPU
    if model.model.device.type != 'cuda':
        print("Error: Script requires GPU. Device is not CUDA. Cancelling script.", flush=True)
        sys.exit(1)
    
    with MilvusWrapper(args.db_path) as db:
        collection_name = 'query_embeddings' if args.type == 'query' else 'image_embeddings'

        for batch in tqdm(dataloader):
            split = args.split
            batch_ids = batch[1]
            
            if args.type == 'query':
                artefact_ids = [dataset.annotations[i]['reference_qid'] for i in batch_ids]
            else:
                artefact_ids = [dataset.candidate_ids[i] for i in batch_ids]

            vectors = model.process(batch[0])

            db.insert_embeddings(
                collection_name=collection_name,
                artefact_ids=artefact_ids,
                batch_ids=batch_ids,
                split=split,
                vectors=vectors,
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path_prefix', type=str)
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--type', choices=['query', 'image'], type=str)
    parser.add_argument('--split', choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_id', choices=["Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B"], type=str)
    parser.add_argument('--db_path', type=Path)
    args = parser.parse_args()    

    embed(args)