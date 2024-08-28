from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import json
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

# Load the .env file
load_dotenv()


#Load Data
with open('diabetes_data_with_vectors', 'r') as f_in:
    diabetes_data_with_vectors = json.load(f_in)
    
api_key = os.getenv('QDRANT_API_KEY')

#Create Client
client = QdrantClient(
    url="https://8999b86c-f8b2-4d60-bdfa-8c68d39daae7.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=api_key,timeout=200
)

#Check If Collection Already Exist or Create
if not client.collection_exists("diabetes"):
    client.create_collection(
        collection_name="diabetes",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    

#Insert Data
client.upsert(
    collection_name="diabetes",
    points=[
        PointStruct(
            id=idx,
            vector=data['qa_text_embeddings'],
            payload={"question": data['question'], "answer": data['answer'],"id": data['id'], "rand_number": idx % 10}
        )
        for idx, data in enumerate(diabetes_data_with_vectors)
    ]
)

collection_name = "diabetes"
count = client.count(collection_name=collection_name)

print(f"Total number of points in the collection '{collection_name}': {count}")