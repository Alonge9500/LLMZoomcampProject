import os
os.environ['HF_HOME'] = '/run/cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from startES import es_client
from sentence_transformers import SentenceTransformer
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

index_name = "diabetes-questions"

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
        result_docs.append(f"Question: {hit['_source']['question']} /n Answer: {hit['_source']['answer']}")

    return result_docs

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
def generate_answer(prompt, context):
    context_str = "\n".join([f"{doc}" for doc in context])
    full_prompt = f"{prompt}\n\nContext:\n{context_str}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt",max_length=512,truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

def rag_function(user_question):
    # Encode the user question
    user_question_embedding = embedding_model.encode(user_question)
    
    # Retrieve similar documents from Elasticsearch
    context = elastic_search_knn('qa_text_embeddings', user_question_embedding)
    
    # Generate the answer using the model
    prompt = f"As a diabetes consultant, provide a comprehensive answer based on the following context."
    answer = generate_answer(prompt, context).rsplit("Answer:", 1)[-1].strip()
    return answer,context


# Streamlit app
st.title("RAG Application")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        with st.spinner("Processing..."):
            answer,context = rag_function(user_question)
            st.write(answer)
            st.write(context)
    else:
        st.warning("Please enter a question.")
