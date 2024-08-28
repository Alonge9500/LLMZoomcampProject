import os
import streamlit as st
import sqlite3
import time
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
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

# Initialize cohere client
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

# Connect to SQLite database
conn = sqlite3.connect('diabetes_chatbot.db')
c = conn.cursor()


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
    start_time = time.time()
    user_question_embedding = embedding_model.encode(user_question)
    context = search_query(user_question_embedding)
    prompt = "As a diabetes consultant, provide a brief answer based on the following context and return only the answer"
    try:
        answer = generate_answer(prompt, context).text
    except:
        answer = generate_answer(prompt, context)
    
    response_time = time.time() - start_time
    
    # Calculate cosine similarity between the user question and LLM response
    llm_embedding = embedding_model.encode(answer)
    cosine_sim = util.pytorch_cos_sim(user_question_embedding, llm_embedding).item()
    
    return answer, response_time, cosine_sim

# Streamlit app
st.title("Diabetes ChatBot Doctor")


if 'messages' not in st.session_state:
    st.session_state.messages = []
    



prompt = st.chat_input('Input Your Message')


for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

    
if 'prompts' not in st.session_state:
    st.session_state.prompts = None

if 'my_state' not in st.session_state:
    st.session_state.my_state = None

if 'response' not in st.session_state:
    st.session_state.response = None
if 'response_time' not in st.session_state:
    st.session_state.response_time = None
if 'cosine_sim' not in st.session_state:
    st.session_state.cosine_sim = None


if prompt:
    st.session_state.prompts = prompt
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    response, response_time, cosine_sim = rag_function(prompt)
    
    st.session_state.response = response
    st.session_state.response_time = response_time
    st.session_state.cosine_sim = cosine_sim
    
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})
    
def feedback():
    try:
        prompt = st.session_state.messages[len(st.session_state.messages)-2]['content']
    except:
        prompt = st.session_state.messages[len(st.session_state.messages)]['content']
    
    
    feedback = st.selectbox('Was this response helpful?', ('👍 Yes', '👎 No'))
    submit = st.button(label='Submit Feedback')
    if submit:
        feedback_str = 'positive' if feedback == '👍 Yes' else 'negative'
        timestamp = datetime.now()

        c.execute("INSERT INTO interactions (user_input, llm_response, cosine_similarity, response_time, feedback, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (prompt,
                   st.session_state.response,
                   st.session_state.cosine_sim,
                   st.session_state.response_time,
                   feedback_str,
                   timestamp))
        conn.commit()

        st.text("Thank you for your feedback!")
        return feedback
    else:
        return None
if len(st.session_state.messages) > 1:
    st.session_state.my_state = feedback()

conn.close()