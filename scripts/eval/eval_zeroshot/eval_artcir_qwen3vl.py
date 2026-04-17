from pymilvus import MilvusClient, DataType
from pathlib import Path
import argparse
from tqdm import tqdm

# Libraries for the model
import torch
from models.qwen3_vl_embedding import Qwen3VLEmbedder
from artcir_thesis.dataset import ArtCIRDataset
from artcir_thesis.submodules.lamra.collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator

# from PIL import Image

# from pathlib import Path
# import multiprocessing
# import pandas as pd
# from tqdm import tqdm
# from collections.abc import Hashable

# def to_milvus_record(args: tuple[Hashable, pd.Series]):
#     index, row = args
#     with Image.open(row['images_label'][0]) as image:
#         vector = clip.embed(images=image).detach().cpu().numpy()[0].tolist()
#     return {
#         'id': index,
#         'artefact': row['artefact'],
#         'vector': vector
#     }

# def init_worker():
#     global clip
#     clip = Clip(model_name='openai/clip-vit-large-patch14')
#     print("Clip running on", clip.device, flush=True)
    

# parser = argparse.ArgumentParser(
#     prog='Images embedding',
# )

# parser.add_argument('--metadata', required=True, type=Path,
#                     help="Path to the metadata file.")

# parser.add_argument('--milvus_path', required=True, type=Path,
#                     help="Path to save the milvus db file.")

# args = parser.parse_args()

# print("Run the embedding task!", flush=True)

# df_metadata = parse_literal_columns(args.metadata)
# notna_mask = df_metadata['images_label'].notna()
# df_metadata = df_metadata.loc[notna_mask, ['artefact', 'images_label']]

# to_milvus_record_args = [(i, row) for i, row in df_metadata.iterrows()]
# with multiprocessing.Pool(processes=1, initializer=init_worker) as pool:
#     data = list(
#         tqdm(
#             pool.imap_unordered(to_milvus_record, to_milvus_record_args),
#             total=len(to_milvus_record_args)
#         )
#     )

# print("The number of vectors to insert:", len(data))

# client = MilvusClient(args.milvus_path.as_posix())
# collection_name = 'images_embedding'

# if client.has_collection(collection_name=collection_name):
#     client.drop_collection(collection_name=collection_name)

# client.create_collection(
#     collection_name=collection_name,
#     dimension=768,  # Clip's output is 768-dimensional
# )

# # BATCH INSERTION LOGIC
# # 1. Redirigir streams ANTES de configurar el logger
# progressbar.streams.wrap_stderr()
# progressbar.streams.wrap_stdout()

# BATCH_SIZE = 1000
# total_inserted = 0
# for i in progressbar.progressbar(range(0, len(data), BATCH_SIZE)):
#     batch = data[i:i + BATCH_SIZE]

#     # Insert the bacth
#     try:
#         res = client.insert(collection_name=collection_name, data=batch)
#         total_inserted += len(batch)
#         print(f"Successfully inserted batch starting at index {i}. Total inserted: {total_inserted}", flush=True)
#     except Exception as e:
#         print(f"Insertion failed for batch starting at index {i}. Error: {e}", flush=True)

# # END OF BATCH INSERTION LOGIC
# client.close()

# print(res)

def load_milvus_database(db_path: Path):
    client = MilvusClient(db_path.as_posix())
    # Create schema
    """
    batch_id: int
    PRIMARY artefact_id: str
    split: {train, val, test}
    type: {query, image}
    query_vector: VECTOR
    image_vector: VECTOR
    """
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name='artefact_id',
        datatype=DataType.VARCHAR,
        max_length=100,
        is_primary=True,
        auto_id=False
    )
    schema.add_field(
        field_name='batch_id',
        datatype=DataType.INT32
    )
    schema.add_field(
        field_name='split',
        datatype=DataType.VARCHAR,
        max_length=5,
    )
    schema.add_field(
        field_name='type',
        datatype=DataType.VARCHAR,
        max_length=5,
    )
    schema.add_field(
        field_name='query_vector',
        datatype=DataType.FLOAT_VECTOR,
        dim=2048
    )
    schema.add_field(
        field_name='image_vector',
        datatype=DataType.FLOAT_VECTOR,
        dim=2048
    )
    
    # 3.3. Prepare index parameters
    index_params = client.prepare_index_params()

    # 3.4. Add indexes
    index_params.add_index(
        field_name="query_vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    index_params.add_index(
        field_name="image_vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    index_params.add_index(
        field_name="split",
        index_type="AUTOINDEX"
    )

    index_params.add_index(
        field_name="type",
        index_type="AUTOINDEX"
    )

    collection_name = 'artcir_embeddings'

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    return client

def qwen_message_collate_fn(batch):
    # batch is a list of dataset items: [(message_dict, idx), ...]
    messages, ids = zip(*batch) # (message_dict1, ...), (idx1, ...)
    return list(messages), list(ids)

def build_dataloader(args):
    dataset = ArtCIRDataset(
        annotation_path_prefix=args.annotation_path_prefix,
        image_path_prefix=args.image_path_prefix,
        type=args.type,
        split=args.split
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qwen_message_collate_fn)

def embed(args):
    # Load the dataset and the model
    dataloader = build_dataloader(args)
    model = Qwen3VLEmbedder(model_name_or_path=args.model_id,
                            torch_dtype=torch.float16
            )
    
    print(model.model.device)

    #client = load_milvus_database(args.db_path)

    for batch in tqdm(dataloader):
        print(batch[0])
        embeddings = model.process(batch[0])
        print(embeddings.shape)
        break

    # for batch in tqdm(dataloader):
    #     batch_messages_dict, batch_ids = batch
        
    #     # Convert dict-of-lists -> list-of-dicts
    #     batch_messages = batch_to_messages(batch_messages_dict, batch_ids)
    #     print(batch_messages_dict, flush=True)
    #     print("Now:")
    #     print(batch_messages, flush=True)
    #     break
        #embeddings = model.process(batch_messages)  # shape: [B, D]
        #print(embeddings.shape, flush=True)



    


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





























#####################################################################################

# from models.qwen3_vl_embedding import Qwen3VLEmbedder
# import numpy as np
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('The device is', device)
# # Define a list of query texts
# queries = [
#     {"text": "A woman playing with her dog on a beach at sunset."},
#     {"text": "Pet owner training dog outdoors near water."},
#     {"text": "Woman surfing on waves during a sunny day."},
#     {"text": "City skyline view from a high-rise building at night."}
# ]

# # Define a list of document texts and images
# documents = [
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
#     {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
# ]

# # Specify the model path
# model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"

# # Initialize the Qwen3VLEmbedder model
# model = Qwen3VLEmbedder(
#     model_name_or_path=model_name_or_path,
#     torch_dtype=torch.float16,
#     #attn_implementation=attn_implementation,
# )
# # We recommend enabling flash_attention_2 for better acceleration and memory saving,
# # model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# # Combine queries and documents into a single input list
# inputs = queries + documents

# # Process the inputs to get embeddings
# embeddings = []
# for index, i in enumerate(inputs):
#     #with torch.inference_mode():
#     embeddings.append(model.process([i]))
#     # if device.type == "cuda":
#     #     torch.cuda.empty_cache()
#     print('Finished the', index, '-th input', flush=True)
# print(embeddings)
# embeddings = torch.cat(embeddings, dim=0)
# # Compute similarity scores between query embeddings and document embeddings
# similarity_scores = (embeddings[:4] @ embeddings[4:].T)

# # Print out the similarity scores in a list format
# print(similarity_scores.tolist())

# # [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]
