# interactive_rag_qa_v5a.py  = = PARTIAL! RAG - R Chain (Fully Local Q&A App)

# GPU-Accelerated Embedding Model Initialization (Now using a local service)

import os
import time
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from local_embeddings import LocalEmbeddings  # Import our custom class

# --- Configuration ---
PERSIST_DB_DIR = "../chroma_db/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
# 2. Main Application Logic
# ----------------------------------------------

def main():
    try:
        # 1. Initialize Local Embeddings
        # The LocalEmbeddings class is now fully self-contained and loads the model on initialization.
        embeddings = LocalEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model client initialized.")

        # 2. Load the Correct Database
        persist_path = get_persist_directory()
        print(f"Loading vector store from: {persist_path}")

        if not os.path.exists(persist_path):
            raise FileNotFoundError(
                f"Database folder not found at {persist_path}. "
                "The metadata file points to a non-existent directory. "
                "Please re-run 'python ingest_pdf.py'."
            )

        # Initialize ChromaDB using our custom local embedding function
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )
        # Create a retriever object
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Setup failed: {e}")
        return
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during setup: {e}")
        return

    print("\n[READY] Application is ready. Ask a question about the PDF.")
    print("Type 'exit' or 'quit' to close the application.")
    print("-" * 60)

    # 3. Interactive Loop
    while True:
        question = input("Q: ")
        if question.lower() in ['exit', 'quit']:
            break

        if not question.strip():
            continue

        start_time = time.time()
        print("\nThinking...")

        try:
            # 4. Retrieval Step (R)
            # Find the most relevant chunks from the local database
            retrieved_docs: List[Document] = retriever.invoke(question)

            retrieval_time = time.time() - start_time
            print(f"Retrieval Time: {retrieval_time:.2f} seconds.")

            # 5. Local Context Output (A)
            # Since we removed the external LLM, we just output the raw context
            print("\n" + "=" * 20 + " RELEVANT CONTEXT (R) " + "=" * 20)

            context = ""
            sources = set()

            for i, doc in enumerate(retrieved_docs):
                page_number = doc.metadata.get('page', 'Unknown')
                source_file = doc.metadata.get('source', 'Unknown')
                sources.add(f"- Source: {source_file} (Page {page_number})")

                print(f"[Chunk {i + 1} from Page {page_number}]:")
                # Clean up and print the content
                cleaned_content = doc.page_content.strip()
                print(f"{cleaned_content[:300]}...\n")  # Print first 300 chars of the chunk

                context += cleaned_content + "\n\n"

            print("=" * 60)

            # 6. Display Sources
            print("\nSources Used:")
            for source in sorted(list(sources)):
                print(source)
            print("-" * 60)

        except Exception as e:
            print(f"\n[ERROR] An error occurred during processing: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()