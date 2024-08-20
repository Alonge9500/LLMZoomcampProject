import os
os.environ['HF_HOME'] = '/run/cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
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

api_key = os.getenv('QDRANT_API_KEY')


client = QdrantClient(
    url="https://8999b86c-f8b2-4d60-bdfa-8c68d39daae7.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=api_key,timeout=200
)


def search_query(query_vector):
    hits = client.search(
    collection_name="diabetes",
    query_vector=query_vector,
    limit=5 )
    
    return hits
    



def generate_answer(prompt, context):
    context_str = "\n".join([f"Q: {doc.payload['question']}\nA: {doc.payload['answer']}" for doc in context])
    full_prompt = f"{prompt}\n\nContext:\n{context_str}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt",max_length=512,truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()
    return answer

def rag_function(user_question):
    # Encode the user question
    user_question_embedding = embedding_model.encode(user_question)
    
    # Retrieve similar documents from Elasticsearch
    context = search_query(user_question_embedding)
    
    # Generate the answer using the model
    prompt = f"As a diabetes consultant, provide a comprehensive answer based on the following context."
    answer = generate_answer(prompt, context)
    return answer

# Streamlit app
st.title("Diabetes ChatBot Doctor")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        with st.spinner("Processing..."):
            answer = rag_function(user_question)
            print(answer)
            st.write(answer)
    else:
        st.warning("Please enter a question.")
