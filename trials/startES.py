from elasticsearch import Elasticsearch
import json
from tqdm.auto import tqdm


es_client = Elasticsearch('http://localhost:9200') 
    
response = es_client.info()
print(response)

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "answer": {"type": "text"},
            "question": {"type": "text"},
            "id": {"type": "keyword"},
            "qa_text_embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "diabetes-questions"
with open('diabetes_data_with_vectors', 'r') as f_in:
    diabetes_data_with_vectors = json.load(f_in)

try:
    es_client.indices.delete(index=index_name, body=index_settings, ignore=400)
    print(f"Index '{index_name}' created or already exists.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    es_client.indices.create(index=index_name, body=index_settings, ignore=400)
    print(f"Index '{index_name}' created or already exists.")
except Exception as e:
    print(f"An error occurred: {e}")
    
### Load Documents

    
for doc in tqdm(diabetes_data_with_vectors):
    es_client.index(index=index_name, document=doc)