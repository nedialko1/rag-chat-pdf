# interactive_rag_qa.py = FULL RAG - R - A  Chain
# GPU-Accelerated Embedding Model Initialization (Now using fully local embedding)

import os
import time
from typing import List

# --- Modern LangChain Imports (Fixed Deprecation Warning) ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# Updated to use the recommended standalone package for Ollama
from langchain_ollama import OllamaLLM

# Import our custom local embedding class (assuming local_embeddings.py is present)
from local_embeddings import LocalEmbeddings

# --- Configuration ---
PERSIST_DB_DIR = "../chroma_db/"
# Ensure this matches the model used in local_embeddings.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
## "mistral"  # Ensure this model is pulled and running in Ollama

# ----------------------------------------------
# 1. Utility Functions
# ----------------------------------------------

def get_persist_directory() -> str:
    """Reads the last ingested database name from the metadata file."""
    METADATA_FILE = os.path.join(PERSIST_DB_DIR, "last_ingested_db.txt")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_FILE}. "
            "Please run 'python ingest_pdf.py' first to create a database."
        )

    with open(METADATA_FILE, 'r') as f:
        db_folder_name = f.read().strip()

    # Construct the full persistence path
    return os.path.join(PERSIST_DB_DIR, db_folder_name)


# ----------------------------------------------
# 2. RAG Prompt Template
# ----------------------------------------------

# This system prompt guides the LLM to synthesize the answer
RAG_PROMPT_TEMPLATE = """
You are an accurate and professional technical assistant. Your task is to answer the user's question
based ONLY on the provided context (the document chunks). Do not use any external knowledge.
If the context does not contain the answer, state clearly that you cannot find the required information.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""


# ----------------------------------------------
# 3. Main Application Logic
# ----------------------------------------------

def main():
    try:
        # 3a. Initialize Local Embeddings
        embeddings = LocalEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model client initialized.")

        # 3b. Load Local LLM (Ollama) - FIXED DEPRECATION WARNING
        # Uses OllamaLLM from langchain_ollama package
        llm = OllamaLLM(model=OLLAMA_MODEL)
        print(f"Local LLM client initialized using Ollama model: {OLLAMA_MODEL}")

        # 3c. Load the Correct Database
        persist_path = get_persist_directory()
        print(f"Loading vector store from: {persist_path}")

        if not os.path.exists(persist_path):
            raise FileNotFoundError(
                f"Database folder not found at {persist_path}. "
                "The metadata file points to a non-existent directory. "
                "Please re-run 'python ingest_pdf.py'."
            )

        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )
        # Create a retriever object that fetches 5 chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Setup failed: {e}")
        return
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during setup: {e}")
        # Note: If Ollama is not running, the error will likely be here.
        print("HINT: Ensure the Ollama service is running and the model ('mistral') is pulled.")
        return

    # Create the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
            PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            | llm
    )

    print("\n[READY] Full Local RAG Application is ready. Ask a question about the PDF.")
    print("Type 'exit' or 'quit' to close the application.")
    print("-" * 60)

    # 4. Interactive Loop
    while True:
        question = input("Q: ")
        if question.lower() in ['exit', 'quit']:
            break

        if not question.strip():
            continue

        start_time = time.time()
        print("\nThinking...")

        try:
            # 5. Retrieval Step (R)
            retrieved_docs: List[Document] = retriever.invoke(question)
            retrieval_time = time.time() - start_time

            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # 6. Augmentation Step (A)
            # Invoke the LLM chain to synthesize the final answer
            synthesis_start = time.time()

            final_answer = rag_chain.invoke({
                "context": context,
                "question": question
            })

            synthesis_time = time.time() - synthesis_start
            total_time = time.time() - start_time

            # 7. Output Results
            print("\n" + "=" * 20 + " SYNTHESIZED ANSWER (RAG-A) " + "=" * 20)
            print(final_answer.strip())
            print("=" * 60)

            # 8. Display Sources and Timing
            sources = set()
            for doc in retrieved_docs:
                page_number = doc.metadata.get('page', 'Unknown')
                source_file = doc.metadata.get('source', 'Unknown')
                sources.add(f"- Source: {source_file} (Page {page_number})")

            print("\nSources Used for Synthesis:")
            for source in sorted(list(sources)):
                print(source)

            print(f"\nTiming Summary:")
            print(f"  Retrieval (Chroma/Embeddings): {retrieval_time:.2f} seconds")
            print(f"  Synthesis (Ollama LLM):        {synthesis_time:.2f} seconds")
            print(f"  Total RAG Time:                {total_time:.2f} seconds")
            print("-" * 60)

        except Exception as e:
            print(f"\n[ERROR] An error occurred during processing: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()