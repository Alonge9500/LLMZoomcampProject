#Load Packages
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

#Load Data
with open('diabetes_data_with_ids', 'r') as file:
    diabetes_data_with_ids = json.load(file)
    
# Create Model Eembeddings
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')


for doc in tqdm(diabetes_data_with_ids):
    question = doc['question']

    doc['question_embeddings'] = embedding_model.encode(question).tolist()


## Save the data with Vectors

with open('diabetes_data_with_question_vectors', 'wt') as f_out:
    json.dump(diabetes_data_with_ids, f_out, indent=2)
    
