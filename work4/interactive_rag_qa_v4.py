# interactive_rag_qa_v4.py  = HF Embeddings + LLM

# GPU-Accelerated Embedding Model Initialization

import requests
import json
import os
import time
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


# ----------------------------------------------
# 1. Custom Class to hit the Local FastAPI Service
# ----------------------------------------------

# 1. Choose a GPU-friendly embedding model (for logging/metadata only)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------
# 2. Load Vector Store and Define Retriever
# ----------------------------------------------

# Define the PDF and the corresponding persistence directory
PDF_FILE_NAME = "../NK_CV_ver_7.pdf"
# Use a directory specific to the file name to enable unique storage/caching
#
PERSIST_DIR = f"chroma_db/TMP"
# PERSIST_DIR = f"./chroma_db/{PDF_FILE_NAME.replace('.', '_')}_chunked"
# PERSIST_DIR = f"./chroma_db/{PDF_FILE_NAME.replace('.', '_')}_full"

# Check if the required vector store directory exists
if not os.path.exists(PERSIST_DIR):
    print("\n--- CRITICAL STARTUP ERROR ---")
    print(f"Vector store for '{PDF_FILE_NAME}' not found at: {PERSIST_DIR}")
    print("Please run the ingestion service first to process the document:")
    print(f"1. Ensure 'embedding_service.py' is running.")
    print("2. Run 'uvicorn ingestion_service:app --reload --port 8001' to process the PDF.")
    exit()

# 2. Configure model to use the GPU
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load Vector Store
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
print(f"Vector store successfully loaded from {PERSIST_DIR}.")

# Define the Retriever
# Keeping k=5 for better context coverage
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LangChain Integration:

from langchain_ollama import OllamaLLM  # This is the new, recommended class
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Initialize Ollama LLM
# RESTORED: Switching back to the more capable llama3.2:3b model.
llm = OllamaLLM(model="llama3.2:3b")

# 2. Define Prompt Template
# RESTORED: Using the "Version #1" prompt structure for improved summarization and gist quality.
prompt = ChatPromptTemplate.from_template("""
    You are an accurate data extraction and summarization assistant. 
    Analyze the provided context and directly answer the user's question by extracting or synthesizing factual information from the context.

    If the context does not contain the answer, politely state that the information is unavailable in the document.

    Context: {context}

    Question: {input}
""")

# 3. Build the RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# NOTE: LLM Cold Start Warm-up has been removed to reduce overhead.
# The model will now load during the first query.

print("LLM and RAG chain successfully configured.")
print("\n" + "=" * 50)
print(f"RAG System is Ready. Querying data from: {PDF_FILE_NAME}")
print("=" * 50 + "\n")

# Run the Query[ies] - Execute the final RAG chain in a loop
while True:
    question = input("Q > ")

    if question.lower() in ["/bye!", "exit", "quit"]:
        print("\n--- Goodbye! ---")
        break

    if not question.strip():
        continue

    print(f"\n--- Processing Query ---")
    try:
        start_time_query = time.time()
        response = retrieval_chain.invoke({"input": question})
        end_time_query = time.time()
        duration = end_time_query - start_time_query

        print("\n--- RAG Answer ---")
        print(response["answer"].strip())
        # Added query duration output
        print(f"(Query duration: {duration:.2f} seconds)")
        print("\n" + "=" * 50 + "\n")
    except ConnectionError as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        print("Please ensure the embedding service is running and try again.")
        break
    except Exception as e:
        print(f"\nAn error occurred during invocation (Is Ollama running?): {e}")
        break  # Exit loop on critical error
