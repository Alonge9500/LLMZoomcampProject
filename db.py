import sqlite3
import time
from datetime import datetime

# Connect to SQLite database (or create it)
conn = sqlite3.connect('diabetes_chatbot.db')
c = conn.cursor()

# Create table with timestamp
c.execute('''CREATE TABLE IF NOT EXISTS interactions
             (user_input TEXT, llm_response TEXT, cosine_similarity REAL, response_time REAL, feedback TEXT, timestamp DATETIME)''')

conn.commit()
