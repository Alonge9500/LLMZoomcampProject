# Diabetes Q&A RAG System
### DATA TALKS LLMZOOMCAMP FINAL PROJECT

## Project Overview

This project focuses on building an end-to-end Retrieval-Augmented Generation (RAG) system for a diabetes Q&A application. The system leverages a diabetes-related dataset, ingests it into a knowledge base, and implements a RAG flow that retrieves relevant information and generates answers to user queries using a Large Language Model (LLM). The project also includes evaluation metrics, a user interface, monitoring features, and detailed instructions for reproducing the results.

## Problem Statement

The aim of this project is to create a system that can provide accurate and relevant answers to diabetes-related questions by combining retrieval and generation techniques. The system is designed to enhance the accuracy of responses by integrating information from a pre-ingested knowledge base and refining the output using a Cohere LLM API.

## Technologies Used

- **LLM:** Cohere API
- **Knowledge Base:** Qdrant and Elasticsearch
- **Monitoring:** Streamlit and Plotly
- **Interface:** Streamlit
- **Ingestion Pipeline:** Python scripts (`ingestion_with_qdrant.py`, `ingestion_with_elasticsearch.py`)

## Project Structure

```plaintext
├── scripts
│   ├── ingestion_with_qdrant.py
│   ├── ingestion_with_elasticsearch.py
│   ├── retrieval_evaluator_qdrant.py
│   ├── retrieval_evaluator_elasticsearch.py
│   ├── rag_evaluator.py
│   ├── dashboard.py
│   ├── diabetesRag.py
├── diabetesRag.py               
├── dashboard.py               
├── README.md
├── requirements.txt
├── docker-compose.yaml
├── .streamlit
│   └── config.toml
├── database.db           
└── images                
│   ├── dashboard.jpg
│   ├── user_feedback.jpg
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Alonge9500/LLMZoomcampProject.git
cd LLMZoomcampProject.git
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2.1 Prepare DATA

To Load and add Index To your DATA and save as CSV file

```bash
python scripts/base_file_indexing.py
```
To Load data with index and create vectors embeddings and save the result as a CSV file

```bash
python scripts/create_vectors.py
```

### 3. Ingest Data into the Knowledge Base

To ingest data into Qdrant:

```bash
python scripts/ingestion_with_qdrant.py
```

To ingest data into Elasticsearch:

```bash
python scripts/ingestion_with_elasticsearch.py
```

### 4. Evaluate Retrieval Performance

To evaluate the Qdrant retrieval:

```bash
python scripts/retrieval_evaluator_qdrant.py
```

**Qdrant Evaluation Results:**
- **HitRATE:** 0.6375
- **MRR:** 0.4593

To evaluate Elasticsearch retrieval (both hybrid and vector-only search):

```bash
python scripts/retrieval_evaluator_elasticsearch.py
```

**Elasticsearch Evaluation Results:**
- **Hybrid Search:**
  - **HitRATE:** 0.64375
  - **MRR:** 0.4605
- **Vector-Only Search:**
  - **HitRATE:** 0.6375
  - **MRR:** 0.4593

### 5. Evaluate RAG Performance

To evaluate the RAG flow:

```bash
python scripts/rag_evaluator.py
```

**RAG Evaluation Results:**
- **Average Cosine Similarity:** 0.85 (across 30 questions)

### 6. Run the Streamlit Application

Launch the Streamlit app to interact with the RAG system:

```bash
streamlit run diabetesRag.py
```
 Or Start with Docker Compose

### 7. Monitoring and User Feedback

To monitor the application and view user feedback, run the dashboard:

```bash
streamlit run monitoring_dashboard.py
```

The monitoring dashboard collects user feedback and some other metrics, store it in the database using sqlite3 and visualizes it using Plotly charts as shown in the image below.
![User Feedback Screenshot](/images/user_feedback.JPG)

## Interface

The application interface is built using Streamlit, providing an easy-to-use platform for users to interact with the RAG system. The interface allows users to ask questions, view answers, and submit feedback.

## Monitoring

The system includes a monitoring feature that collects user feedback and displays it on a dashboard created with Streamlit and Plotly. The dashboard includes various charts to track system performance and user interactions.
Link to Monitoring Dashboard -> https://monitoringdashboard.streamlit.app/

![Dashboard Screenshot](/images/dashboard.JPG)

## Reproducibility

This project is fully reproducible. All necessary instructions for running the code, from setting up the environment to executing the scripts, are provided. The dataset used is accessible, and the code is organized for easy execution. 

### Dockerization

The project includes a `docker-compose` for managing dependencies.


- **Hybrid Search:** Implemented and evaluated


## Deployment

The application can be deployed to the cloud using Streamlit Cloud: https://llmzoomcampprojectdiabetesapp.streamlit.app/
![Cloud Deployment Screenshot](/images/cloud.JPG)

## Conclusion

This project put to use all the knowledege gathered in LLMZOOMCAMP 2024 by DATATALKS, it demonstrates the effective use of RAG techniques to build a robust Q&A system for diabetes-related queries. It includes comprehensive evaluation metrics, a user-friendly interface, and a detailed monitoring dashboard, making it a valuable tool for both end-users and developers.
