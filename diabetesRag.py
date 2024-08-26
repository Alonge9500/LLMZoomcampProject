import os
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import cohere

st.set_page_config(
    page_title="Diabetes App",
    page_icon=":sparkles:", 
    layout="centered",  
    initial_sidebar_state="expanded",
)
# Load environment variables
load_dotenv()

# Initialize cohere clientCOHERE_API_KEY
cohere_api_key = os.getenv('COHERE_API_KEY')

co = cohere.Client(cohere_api_key)

# Initialize models
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

api_key = os.getenv('QDRANT_API_KEY')

client = QdrantClient(
    url="https://8999b86c-f8b2-4d60-bdfa-8c68d39daae7.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=api_key, 
    timeout=200
)

def search_query(query_vector):
    hits = client.search(
        collection_name="diabetes",
        query_vector=query_vector,
        limit=5
    )
    return hits

def generate_answer(prompt, context):
    if context[0].score < 0.4:
        return 'I\'m a diabetes chat bot assistant ask questions related to diabetes alone....Thank You.'
    else:
        context_str = "\n".join([f"Question: {doc.payload['question']}\nAnswer: {doc.payload['answer']}" for doc in context])
        full_prompt = f"{prompt}\n\nContext:\n{context_str}\n\nAnswer:"
        response = co.chat(message=full_prompt)
        return response

def rag_function(user_question):
    user_question_embedding = embedding_model.encode(user_question)
    context = search_query(user_question_embedding)
    prompt = "As a diabetes consultant, provide a brief answer based on the following context and return only the answer"
    try:
        answer = generate_answer(prompt, context).text
    except:
        answer = generate_answer(prompt, context)
    return answer

# Streamlit app
st.title("Diabetes ChatBot Doctor")

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Input Your Message')
if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    response = rag_function(prompt)
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})