import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon=":sparkles:", 
    layout="wide",  
    initial_sidebar_state="expanded",
)

st.header('Rag (Diabetes APP) Monitoring Dashboard')
# Load data from SQLite database
conn = sqlite3.connect('diabetes_chatbot.db')
query = "SELECT * FROM interactions"
df = pd.read_sql(query, conn)
conn.close()

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a 2-column layout
col1, col2, col3 = st.columns(3)

# Plot 1: Distribution of Feedback
with col1:
    fig1 = px.histogram(df, x='feedback', title='Distribution of Feedback')
    st.plotly_chart(fig1, use_container_width=True)

    
# Plot 2: Cosine Similarity Over Time
with col2:
    fig8 = px.line(df, x='timestamp', y='cosine_similarity', title='Cosine Similarity Over Time')
    st.plotly_chart(fig8, use_container_width=True)
    
# Plot 3: Average Cosine Similarity by Feedback
with col1:
    fig10 = px.box(df, x='feedback', y='cosine_similarity', title='Average Cosine Similarity by Feedback')
    st.plotly_chart(fig10, use_container_width=True)

# Plot 4: Average Response Time by Feedback
with col1:
    fig4 = px.box(df, x='feedback', y='response_time', title='Average Response Time by Feedback')
    st.plotly_chart(fig4, use_container_width=True)

# Plot 5: Cosine Similarity vs. Response Time
with col2:
    fig5 = px.scatter(df, x='cosine_similarity', y='response_time', color='feedback', title='Cosine Similarity vs. Response Time')
    st.plotly_chart(fig5, use_container_width=True)

# Plot 6: Number of Interactions Over Time
with col3:
    fig6 = px.histogram(df, x='timestamp', title='Number of Interactions Over Time')
    st.plotly_chart(fig6, use_container_width=True)

# Plot 7: Feedback Over Time
with col1:
    fig7 = px.histogram(df, x='timestamp', color='feedback', title='Feedback Over Time', barmode='group')
    st.plotly_chart(fig7, use_container_width=True)

# Plot 8: Response Time Distribution
with col2:
    fig2 = px.histogram(df, x='response_time', nbins=20, title='Response Time Distribution')
    st.plotly_chart(fig2, use_container_width=True)

# Plot 9: Response Time Over Time
with col3:
    fig9 = px.line(df, x='timestamp', y='response_time', title='Response Time Over Time')
    st.plotly_chart(fig9, use_container_width=True)



# Plot 10: Cosine Similarity Distribution
with col3:
    fig3 = px.histogram(df, x='cosine_similarity', nbins=20, title='Cosine Similarity Distribution')
    st.plotly_chart(fig3, use_container_width=True)
