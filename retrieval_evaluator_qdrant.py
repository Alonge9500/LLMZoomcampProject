#Import Liraries

import pandas as pd
from tqdm.auto import tqdm
import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import cohere

# Load the .env file
load_dotenv()

#Instantiate QDrant 
api_key = os.getenv('QDRANT_API_KEY')


client = QdrantClient(
    url="https://8999b86c-f8b2-4d60-bdfa-8c68d39daae7.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=api_key,timeout=200
)

#Load Data
data = pd.read_csv('retrieval_evaluation.csv')
retrieval_evaluation_dict = data.to_dict(orient = 'records')
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')



#Define Evaluation Functions

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
#Define Search Query
def search_query(query_vector):
    hits = client.search(
        collection_name="diabetes",
        query_vector=query_vector,
        limit=5
    )
    return hits

#Evaluate
def retrieval_evaluator(data_dictionary):
    relevance_total = []

    for question in tqdm(data_dictionary):
        question_id = question['id']
        vector = embedding_model.encode(question['question'])
        results = search_query(vector)
        relevance = [d.payload['id'] == question_id for d in results]
        relevance_total.append(relevance)
        
    return hit_rate(relevance_total),mrr(relevance_total)

hitrate,mrr = retrieval_evaluator(retrieval_evaluation_dict)
print(f'Hit Rate Is : {hitrate}')
print(f'MRR Is : {mrr}')