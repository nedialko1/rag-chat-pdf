# **RAG Q&A System for CV Document**
This is a Retrieval-Augmented Generation (RAG) system designed to perform highly accurate question-answering on a single PDF document (NK_CV_ver_7.pdf).
This system uses a Full Document Load strategy (no chunking) to ensure the Large Language Model (LLM) sees the entire context for optimal summarization and synthesis quality, specifically utilizing the highly capable llama3.2:3b model.
# **The Three-Component Architecture (The "3-Therapy" Model)**
The application requires three distinct processes to be operational: the Embedding Service, the Ingestion Step, and the Interactive Q&A application.
## **Prerequisites**
Python Environment: Python 3.9+ and a virtual environment (.venv) initialized.
Ollama: Ensure the Ollama server is installed and running on your system.
Required LLM/Embedding Models: Pull the following models using the Ollama CLI:


``` bash
# 1. LLM for Generation (Q&A)
ollama pull llama3.2:3b
# 2. Embedding Model (used by the FastAPI service)
ollama pull all-minilm
# **Install Dependencies**: Install the required Python packages:
pip install -r requirements.txt
``` 

## **Startup Workflow (Three Steps)**
You need three separate terminal windows to run the full application suite:
### **Step 1: Start the Embedding Service (Terminal 1)**
This service handles the core vector creation (embeddings) for both document content and user queries. It must be running on port 8000.
``` bash
# Terminal 1: Start the Embedding Service on port 8000
python embedding_service.py
```
### **Step 2: Ingest the Document Data (Terminal 2 - Run Once)**
This process reads the PDF, uses the running embedding service to create a single vector for the entire document, and saves it to the persistent Chroma Vector Store. This step overwrites any previous data.
``` bash
# Terminal 2: Run the Ingestion Script
python ingestion_service_full_load.py
```

### **Step 3: Start the Interactive Q&A (Terminal 3 - Main Application)**
Once ingestion is complete, start the main application. This connects the retriever to ChromaDB and utilizes the llama3.2:3b LLM for generation.
``` bash
# Terminal 3: Start the RAG Application
python interactive_rag_qa.py
```

## **Performance Note:** 
The very first query will load the llama3.2:3b model and take the longest time (100-200 seconds). Subsequent queries should be much faster (~15-20 seconds). Delays between queries may cause the model to be swapped out of memory, potentially leading to another delay on the next question.

