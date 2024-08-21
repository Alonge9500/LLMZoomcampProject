import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Initialize models
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

# Use TFLAN model (flan-t5-base)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


api_key = os.getenv('QDRANT_API_KEY')

client = QdrantClient(
    url="https://8999b86c-f8b2-4d60-bdfa-8c68d39daae7.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key=api_key, timeout=200
)

def search_query(query_vector):
    hits = client.search(
        collection_name="diabetes",
        query_vector=query_vector,
        limit=5
    )
    return hits

def generate_answer(prompt, context):
    context_str = "\n".join([f"Question: {doc.payload['question']}\nAnswer: {doc.payload['answer']}" for doc in context])
    full_prompt = f"{prompt}\n\nContext:\n{context_str}\n\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = model.generate(**inputs, max_new_tokens=500)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def rag_function(user_question):
    # Encode the user question
    user_question_embedding = embedding_model.encode(user_question)
    
    # Retrieve similar documents from Qdrant
    context = search_query(user_question_embedding)
    
    # Generate the answer using the model
    prompt = "As a diabetes consultant, provide a comprehensive answer based on the following context."
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
