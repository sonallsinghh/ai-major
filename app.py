import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000"

st.title("AI Assistant with RAG & Agents 🚀")

# Tab selection
tab = st.sidebar.radio("Select Action", ["Ask AI", "Weather Info", "News Search", "Upload PDF"])

# Ask AI
if tab == "Ask AI":
    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        try:
            response = requests.post(f"{API_URL}/ask/", json={"query": query})
            response.raise_for_status()  # Raise an exception for bad status codes
            st.write("AI Response:", response.json().get("response", "No response received"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {str(e)}")

# Weather Info
elif tab == "Weather Info":
    city = st.text_input("Enter city name:")
    if st.button("Get Weather"):
        try:
            # Creating the proper messages format for the agent endpoint
            messages = [{"content": f"What is the weather in {city}?", "type": "human"}]
            response = requests.post(f"{API_URL}/agent/", json={"messages": messages})
            response.raise_for_status()
            st.write("Weather Info:", response.json().get("response", "No response received"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {str(e)}")

# News Search
elif tab == "News Search":
    topic = st.text_input("Enter topic:")
    if st.button("Get News"):
        try:
            # Creating the proper messages format for the agent endpoint
            messages = [{"content": f"Give me the latest news about {topic}", "type": "human"}]
            response = requests.post(f"{API_URL}/agent/", json={"messages": messages})
            response.raise_for_status()
            st.write("News:", response.json().get("response", "No response received"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {str(e)}")

# Upload PDF
elif tab == "Upload PDF":
    pdf_url = st.text_input("Enter PDF URL:")
    if st.button("Upload & Process"):
        try:
            response = requests.post(f"{API_URL}/ingest/", json={"pdf_url": pdf_url})
            response.raise_for_status()
            st.success(response.json().get("message", "PDF processed successfully"))
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading PDF: {str(e)}")