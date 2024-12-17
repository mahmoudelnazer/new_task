# RAG Document Chat System

## Overview

The **RAG Document Chat System** is an AI-powered solution for interacting with uploaded documents. Users can upload documents, which are then indexed and stored. The system allows users to query these documents using natural language, providing intelligent and context-aware responses.

### Key Features
- Upload multiple documents (PDF, DOCX, TXT)
- Query documents and receive relevant responses
- Health check for monitoring system status
- Streamlit frontend for easy interaction
- FastAPI backend for document processing and query handling

### Architecture
The system is composed of two main components:
1. **Frontend**: Built using Streamlit for easy user interaction.
2. **Backend**: FastAPI handles document upload, query processing, and system health.

### Requirements
- **Python 3.x**
- **Docker**
- **Azure OpenAI API Key** for generating embeddings and processing queries.

### Setting Up the Environment
1. Clone this repository:

   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
Create a .env file with the following variables:

bash
Copy code
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
CHUNK_SIZE=1024
CHUNK_OVERLAP=20
MODEL_NAME=gpt-4o
EMBEDDING_MODEL_NAME=text-embedding-ada-002
MAX_TOKENS=2000
TEMPERATURE=0.7
Build and start the containers using Docker Compose:

bash
Copy code
docker-compose up --build
Access the application:

Frontend: http://localhost:8501
Backend: http://localhost:8000
API Endpoints
POST /upload: Upload documents for indexing.
Body: [{ "filename": "document.pdf", "content": "base64-encoded-content", "mimetype": "application/pdf" }]
POST /query: Query the uploaded documents.
Body: {"query": "What is the document about?"}
GET /health: Check the health status of the system.
Docker Compose Configuration
This project uses Docker Compose to run the backend and frontend services in containers. Both services are connected to the same network for communication.

Conclusion
The RAG Document Chat System provides a seamless experience for querying documents with AI-powered responses. The system is highly configurable and can be extended for more complex use cases.
