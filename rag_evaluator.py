#Import Libraries

import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import cohere
from tqdm.auto import tqdm


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

#Load DAta
with open('diabetes_data_with_vectors', 'r') as f:
    data = json.load(f)
    

# Select 30 random entries
sampled_data = random.sample(data, 30)

# Prepare lists to store data
ids = []
questions = []
original_answers = []
llm_answers = []
cosine_similarities = []

# Process each sampled entry
for entry in tqdm(sampled_data):
    question = entry['question']
    original_answer = entry['answer']
    question_id = entry['id']
    
    # Generate LLM answer
    llm_answer = rag_function(question)
    
    # Compute embeddings
    original_embedding = embedding_model.encode(original_answer)
    llm_embedding = embedding_model.encode(llm_answer)
    
    # Compute cosine similarity
    similarity = cosine_similarity([original_embedding], [llm_embedding])[0][0]
    
    # Store data
    ids.append(question_id)
    questions.append(question)
    original_answers.append(original_answer)
    llm_answers.append(llm_answer)
    cosine_similarities.append(similarity)

# Create DataFrame
df = pd.DataFrame({
    'ID': ids,
    'Question': questions,
    'Original Answer': original_answers,
    'LLM Answer': llm_answers,
    'Cosine Similarity': cosine_similarities
})

print(df)
print(f"The Average Cosine Similarity fo the 30 samples is {df['Cosine Similarity'].mean()}")

# Save to a CSV file
df.to_csv('llm_comparison_results.csv', index=False)

print("Process completed. Results saved to 'llm_comparison_results.csv'.")