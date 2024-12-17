
import os
import uvicorn
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import logging
import requests
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.text_splitter import SentenceSplitter
from dotenv import load_dotenv
import os
import io
import json

# Load environment variables from the .env file
load_dotenv()


class FileUpload(BaseModel):
    filename: str
    content: str  # Base64 encoded content
    mimetype: Optional[str] = None


class QueryRequest(BaseModel):
    query: str


class RAGChatbot:
    _instance = None
    _index_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Azure OpenAI Configuration
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = "2024-02-15-preview"
        self.index = None

        # Load configuration from environment
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1024))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 20))
        self.model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002')
        self.max_tokens = int(os.getenv('MAX_TOKENS', 2000))
        self.temperature = float(os.getenv('TEMPERATURE', 0.7))

        # Validate environment variables
        if not self.api_key or not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in the environment.")

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            self.llm = AzureOpenAI(
                model=self.model_name,
                deployment_name="gpt4o",
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            self.embed_model = AzureOpenAIEmbedding(
                model=self.embedding_model_name,
                deployment_name="text-embedding-ada-002",
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
            )
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model

            # Initialize the text splitter here
            self.text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            # Try to load existing index
            self._load_existing_index()
        except (requests.exceptions.RequestException, ValueError) as e:
             self.logger.error(f"Initialization error: {str(e)}")
             raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise HTTPException(status_code=500, detail="Initialization Error")



    def _load_existing_index(self):
        try:
            if os.path.exists("./data/index"):
                storage_context = StorageContext.from_defaults(persist_dir="./data/index")
                self.index = load_index_from_storage(storage_context)
                self._index_initialized = True
                self.logger.info("Successfully loaded existing index")
            else:
                self.logger.info("No existing index found.")
        except Exception as e:
            self.logger.warning(f"Could not load existing index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not load existing index: {str(e)}")


    async def upload_documents(self, files):
        try:
            # Test embedding endpoint
            test_payload = {"input": "test"}
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}
            test_url = f"{self.azure_endpoint}/openai/deployments/text-embedding-ada-002/embeddings?api-version={self.api_version}"
            response = requests.post(test_url, json=test_payload, headers=headers)
            response.raise_for_status()
            self.logger.info("Embedding endpoint is accessible. Processing files...")

            # Process documents and create an index
            all_documents = []
            for file_upload in files:
                file_content = base64.b64decode(file_upload.content)
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_upload.filename)[1]) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name
                    try:
                        reader = SimpleDirectoryReader(input_files=[tmp_file_path])
                        documents = reader.load_data()
                        all_documents.extend(documents)
                    finally:
                        os.remove(tmp_file_path)

            if all_documents:
                self.index = VectorStoreIndex.from_documents(
                    all_documents, text_splitter=self.text_splitter
                )
                try:
                    # Persist the index
                    self.index.storage_context.persist("./data/index")
                    self._index_initialized = True
                    return {"message": "Documents processed and indexed successfully!"}
                except Exception as e:
                    self.logger.error(f"Error persisting index: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error persisting index: {str(e)}")

            else:
                return {"message": "No documents found or processed."}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Connection test error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding endpoint unreachable: {str(e)}")
        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")


    async def query_documents(self, query: str) -> Optional[str]:
        if not self._index_initialized or self.index is None:
            raise HTTPException(status_code=400, detail="No documents have been processed yet")
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, response_mode="compact"
            )
            response = query_engine.query(query)
            return response.response
        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


# FastAPI App Setup
app = FastAPI(
    title="RAG Document Chat API",
    description="API for Retrieval-Augmented Generation Document Chatting",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Chatbot
rag_chatbot = RAGChatbot()


@app.post("/upload")
async def upload_documents(files: List[FileUpload]):
    """Upload documents for indexing"""
    try:
        return await rag_chatbot.upload_documents(files)
    except HTTPException as e:
         return {"message": e.detail}


@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query uploaded documents"""
    try:
        response = await rag_chatbot.query_documents(request.query)
        return {"response": response}
    except HTTPException as e:
       return {"message": e.detail}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "index_initialized": rag_chatbot._index_initialized}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)