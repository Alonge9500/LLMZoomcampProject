import os
os.environ['HF_HOME'] = '/run/cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm

# Initialize models
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)



index_name = 'qa_text_embeddings'
# Initialize Elasticsearch client
es_client = Elasticsearch('http://localhost:9200')

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

index_name = "diabetes-questions-main"

# Create the index and ignore if it already exists
try:
    es_client.indices.create(index=index_name, body=index_settings, ignore=400)
    print(f"Index '{index_name}' created or already exists.")
except Exception as e:
    print(f"An error occurred: {e}")
### Load Documents
with open('diabetes_data_with_vectors', 'r') as f_in:
    diabetes_data_with_vectors = json.load(f_in)
    
for doc in tqdm(diabetes_data_with_vectors):
    es_client.index(index=index_name, document=doc)

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
        result_docs.append((hit['_source']['id'], hit['_source']))

    return result_docs

def generate_answer(prompt, context):
    context_str = "\n".join([f"Q: {doc['question']}\nA: {doc['answer']}" for _, doc in context])
    full_prompt = f"{prompt}\n\nContext:\n{context_str}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt",max_length=512,truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def rag_function(user_question):
    # Encode the user question
    user_question_embedding = embedding_model.encode(user_question)
    
    # Retrieve similar documents from Elasticsearch
    context = elastic_search_knn('qa_text_embeddings', user_question_embedding)
    
    # Generate the answer using the model
    prompt = f"As a diabetes consultant, provide a comprehensive answer based on the following context."
    answer = generate_answer(prompt, context)
    return answer

# Streamlit app
st.title("RAG Application")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        with st.spinner("Processing..."):
            answer = rag_function(user_question)
            print(answer)
            st.write(answer)
    else:
        st.warning("Please enter a question.")
