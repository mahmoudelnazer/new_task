
import streamlit as st
import requests
import base64
from typing import List
import io
import json
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Use environment variable with default


def encode_file_to_base64(file: io.BytesIO) -> str:
    """Encode file content to base64"""
    return base64.b64encode(file.getvalue()).decode('utf-8')


def upload_documents(files: List[io.BytesIO]):
    """Upload documents to the RAG API with base64 encoding"""
    try:
        file_uploads = [
            {
                "filename": file.name,
                "content": encode_file_to_base64(file),
                "mimetype": file.type
            } for file in files
        ]
        response = requests.post(
            f"{API_BASE_URL}/upload", json=file_uploads, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload error: {e}")
        if e.response:
            try:
                error_content = e.response.json()
                st.error(f"Error from server: {error_content}")
            except json.JSONDecodeError:
                st.error(f"Error from server: {e.response.text}")
        return None


def query_documents(query: str):
    """Send query to RAG API and get response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query", json={"query": query}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query error: {e}")
        if e.response:
            try:
                error_content = e.response.json()
                st.error(f"Error from server: {error_content}")
            except json.JSONDecodeError:
                st.error(f"Error from server: {e.response.text}")
        return None


def check_system_health():
    """Check backend system health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except:
        return None


def main():
    st.set_page_config(
        page_title="Document RAG Chat",
        page_icon="📄",
        layout="wide"
    )

    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📄 RAG Document Chat")

    # Check system health
    health_status = check_system_health()
    if health_status:
        if health_status.get("index_initialized"):
            st.sidebar.success("System is ready with initialized index")
        else:
            st.sidebar.warning("System is ready but no documents are indexed")
    else:
        st.sidebar.error("Cannot connect to the backend system")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose documents", type=['txt', 'pdf', 'docx'], accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Uploading and processing documents..."):
                    result = upload_documents(uploaded_files)
                    if result:
                        st.success(result.get('message', 'Documents processed successfully!'))
                        st.session_state['documents_processed'] = True
                    else:
                        st.error("Failed to process documents")
                        st.session_state['documents_processed'] = False

    # Chat Section
    st.header("💬 Chat with Your Documents")

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'query_processing' not in st.session_state:
        st.session_state['query_processing'] = False

    # Display chat history
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            st.markdown(f"<div class='chat-message user-message'>**You:** {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message assistant-message'>**Assistant:** {message['content']}</div>", unsafe_allow_html=True)

    # Chat input and "Send" button
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input(
            "Enter your query:",
            placeholder="Ask a question about your documents...",
            key="user_input",
            disabled=st.session_state['query_processing']  # Disable input while processing
        )
    with col2:
        if st.button("Send", disabled=st.session_state['query_processing']):
            if user_query.strip():  # Ensure query is not empty
                st.session_state['query_processing'] = True  # Lock input and button
                with st.spinner("Generating response..."):
                    response = query_documents(user_query)
                    if response:
                        # Append user query and assistant response to chat history
                        st.session_state['chat_history'].append({'role': 'user', 'content': user_query})
                        st.session_state['chat_history'].append({
                            'role': 'assistant',
                            'content': response.get('response', 'No response generated')
                        })
                    else:
                        st.error("Failed to fetch a response.")
                st.session_state['query_processing'] = False  # Unlock input and button
                st.rerun()  # Refresh UI to display response
            else:
                st.warning("Please enter a query before sending.")

    # Add a clear chat button
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state['chat_history'] = []
        st.rerun()

if __name__ == "__main__":
    if 'documents_processed' not in st.session_state:
        st.session_state['documents_processed'] = False
    main()
