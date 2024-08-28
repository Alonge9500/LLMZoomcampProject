import pandas as pd
import hashlib
from collections import defaultdict
from tqdm.auto import tqdm
import json

diabetes_raw_df = pd.read_parquet("hf://datasets/abdelhakimDZ/diabetes_QA_dataset/data/train-00000-of-00001.parquet")
diabetes_raw_df.head()

diabetes_raw_df.drop_duplicates(inplace=True)

# Convert Dataframe to dictionary
diabetes_raw_dict = diabetes_raw_df.to_dict(orient='records')

#Generate ID
def generate_document_id(doc):
    # combined = f"{doc['course']}-{doc['question']}"
    combined = f"{doc['question']}-{doc['answer'][:15]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id

for doc in tqdm(diabetes_raw_dict):
    doc['id'] = generate_document_id(doc)
    
hashes = defaultdict(list)
for doc in diabetes_raw_dict:
    doc_id = doc['id']
    hashes[doc_id].append(doc)
    
## Save the data with ID

with open('diabetes_data_with_ids', 'wt') as f_out:
    json.dump(diabetes_raw_dict, f_out, indent=2)