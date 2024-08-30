#Import Libraries
from elasticsearch import Elasticsearch
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd


#Strat Connection
es_client = Elasticsearch('http://localhost:9200') 


#Index Settings

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
try:
    es_client.indices.create(index=index_name, body=index_settings, ignore=400)
    print(f"Index '{index_name}' created or already exists.")
except Exception as e:

    ### Load Documents
    with open('diabetes_data_with_vectors', 'r') as f_in:
        diabetes_data_with_vectors = json.load(f_in)

    # Re index Data   
    for doc in tqdm(diabetes_data_with_vectors):
        es_client.index(index=index_name, document=doc)
    

    
#Embedding Model
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')


## Elastic Search KNN Vector Only Function
def elastic_search_knn(field, vector):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
    }

    search_query = {
        "knn": knn,
        "_source": ["answer", "question", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )

    result_docs = []

    for hit in es_results['hits']['hits']:
 
        result_docs.append({"Question": hit['_source']['question'] , "Answer": hit['_source']['answer'], "ID": hit['_source']['id']})

    return result_docs

# Hybrid Search
def hybrid_search(query_text, query_vector, top_k=5):
    script_query = {
        "script_score": {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"question": query_text}},
                        {"match": {"answer": query_text}}
                    ]
                }
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'qa_text_embeddings') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    response = es_client.search(
        index=index_name,
        body={
            "size": top_k,
            "query": script_query
        }
    )
    result_docs = []

    for hit in response['hits']['hits']:
 
        result_docs.append({"Question": hit['_source']['question'] , "Answer": hit['_source']['answer'], "ID": hit['_source']['id']})

    return result_docs


data = pd.read_csv('retrieval_evaluation.csv')

retrieval_evaluation_dict = data.to_dict(orient = 'records')

## Define evaluator Functions
def hit_rate_function(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr_function(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def retrieval_evaluator_hybrid(data_dictionary):
    relevance_total = []

    for question in tqdm(data_dictionary):
        question_id = question['id']
        vector = embedding_model.encode(question['question'])
        results = hybrid_search(question['question'], vector)
        relevance = [d['ID'] == question_id for d in results]
        relevance_total.append(relevance)
        
    return hit_rate_function(relevance_total),mrr_function(relevance_total)


def retrieval_evaluator_vector(data_dictionary):
    relevance_total = []

    for question in tqdm(data_dictionary):
        question_id = question['id']
        vector = embedding_model.encode(question['question'])
        results = elastic_search_knn('qa_text_embeddings', vector)
        relevance = [d['ID'] == question_id for d in results]
        relevance_total.append(relevance)

    hit_rate_value = hit_rate_function(relevance_total)
    mrr_value = mrr_function(relevance_total)
    return hit_rate_value, mrr_value

hitrate_hybrid,mrr_hybrid = retrieval_evaluator_hybrid(retrieval_evaluation_dict)
hitrate_vector,mrr_vector = retrieval_evaluator_vector(retrieval_evaluation_dict)


#Display Results
print(f'Hit Rate for hybrid search Is : {hitrate_hybrid}')
print(f'MRR for hybrid search : {mrr_hybrid}')
print('.....................................................................')
print(f'Hit Rate for vector search Is : {hitrate_vector}')
print(f'MRR for vector search : {mrr_vector}')