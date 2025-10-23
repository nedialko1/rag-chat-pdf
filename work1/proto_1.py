import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION ---
PDF_PATH = "../PDF_work/NK_CV_ver_6.pdf"
OLLAMA_MODEL = "llama2"  # Ensure this model is pulled in Ollama (e.g., 'ollama pull llama2')


# --- 2. DATA PREPARATION (Load and Chunk) ---
def load_and_split_documents(pdf_path):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# --- 3. EMBEDDING & VECTOR STORE (Chroma) ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings  # Or any other Embeddings class

def create_vector_store(
        chunks: list[Document],
        embedding_function,
        persist_dir: str = "./chroma_db"
) -> Chroma:
    """
    Creates and persists a Chroma vector store from a list of LangChain Documents.

    Args:
        chunks: A list of LangChain Document objects (text chunks with metadata).
        embedding_function: The embedding model object (e.g., OpenAIEmbeddings).
        persist_dir: The directory path to save the Chroma database files.

    Returns:
        A Chroma vector store instance.
    """

    # 1. Create the vector store from the list of documents
    # The .from_documents() method automatically:
    # - Calculates embeddings for each document using the provided function.
    # - Stores the embeddings, the original document content, and the metadata.
    # - Persists the data to the specified directory.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_dir
    )

    # Optional: Confirm the count
    print(f"Successfully created Chroma store with {vector_store._collection.count()} documents.")

    return vector_store


# --- 4. RAG CHAIN ---
def setup_rag_chain(vector_store):
    # Initialize the local LLM via Ollama
    llm = Ollama(model=OLLAMA_MODEL)

    # Define a custom prompt for the LLM
    prompt_template = """Use the following pieces of context to answer the user's question concisely. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False  # Set to True to see the context used
    )
    return qa_chain


# --- 5. EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file '{PDF_PATH}' not found. Please place your document in the project folder.")
    else:
        print("Starting RAG setup...")

        # 1. Load and Split
        document_chunks = load_and_split_documents(PDF_PATH)
        print(f"Loaded {len(document_chunks)} chunks.")

        # 2. Embed and Store
        vector_db = create_vector_store(document_chunks)
        print("Vector store created with FAISS and Ollama Embeddings.")

        # 3. Setup RAG Chain
        rag_chain = setup_rag_chain(vector_db)
        print(f"RAG chain set up with {OLLAMA_MODEL} LLM.")

        # 4. Ask a question
        query = "What is the main topic of the document?"
        print(f"\nUser Query: {query}")

        # Invoke the RAG chain
        response = rag_chain.invoke(query)

        print("\n--- LLM Response ---")
        print(response['result'])
        print("--------------------")