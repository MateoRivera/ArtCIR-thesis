from pymilvus import MilvusClient, DataType
from pathlib import Path
import numpy as np
import torch

class MilvusWrapper:
    def __init__(self, db_path: Path):
        self.client = MilvusClient(db_path.as_posix())
        self.query_collection = "query_embeddings"
        self.image_collection = "image_embeddings"
        self._ensure_collections()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_collections(self):
        if not self.client.has_collection(self.query_collection):
            self._create_query_collection()
        if not self.client.has_collection(self.image_collection):
            self._create_image_collection()

    def _create_query_collection(self):
        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("artefact_id", DataType.VARCHAR, max_length=100)
        schema.add_field("batch_id", DataType.INT32)
        schema.add_field("split", DataType.VARCHAR, max_length=5)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2048)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        self.client.create_collection(
            collection_name=self.query_collection,
            schema=schema,
            index_params=index_params
        )

    def _create_image_collection(self):
        schema = MilvusClient.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("artefact_id", DataType.VARCHAR, max_length=100)
        schema.add_field("batch_id", DataType.INT32)
        schema.add_field("split", DataType.VARCHAR, max_length=5)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2048)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        self.client.create_collection(
            collection_name=self.image_collection,
            schema=schema,
            index_params=index_params
        )

    def insert_embeddings(self, 
                          collection_name: str, 
                          artefact_ids: list,
                          batch_ids: list, 
                          split: str,
                          vectors
        ):
        # vectors: torch.Tensor float16
        vectors = vectors.to(torch.float32)          # convertir a float32
        vectors = vectors.cpu().numpy()              # a numpy
        vectors = vectors.astype("float32")          # asegurar dtype

        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalización
        # Convert to list format for Milvus
        vectors_list = vectors.tolist()

        # Build list of dictionaries (one per entity)
        data = []
        for i in range(len(batch_ids)):
            entity = {
                "artefact_id": artefact_ids[i],
                "batch_id": batch_ids[i],
                "split": split,
                "vector": vectors_list[i]
            }

            data.append(entity)

        self.client.insert(collection_name, data)

    def search(self, collection_name: str, query_vector, top_k=10, **kwargs):
        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector /= np.linalg.norm(query_vector)
        return self.client.search(
            collection_name=collection_name, 
            data=[query_vector.tolist()],
            anns_field='vector',
            limit=top_k,
            **kwargs
        )

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()
