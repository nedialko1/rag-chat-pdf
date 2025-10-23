# GPU-Accelerated Embedding Model Initialization

from langchain_huggingface import HuggingFaceEmbeddings

# 1. Choose a GPU-friendly embedding model
# all-MiniLM-L6-v2 is small, fast, and a great default for a modest GPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Configure model to use the GPU
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print(f"Embedding model loaded on device: {embeddings.model_kwargs['device']}")

# Data Loading and Chunking: Standard RAG procedure to prepare your documents.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Documents
# Rather poorly fitted for AI processing: NK_CV_ver_6
# Much better (no repeats in footer:
#   : NK_CV_ver_6_AI_friendly.pdf // Remember to reset the DB !!!
# BEST this far :)
#   NK_CV_ver_7.pdf
loader = PyPDFLoader("NK_CV_ver_7.pdf") # Replace with your file
documents = loader.load()

# 2. Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

# Create/Load Vector Store (ChromaDB)
# Use the GPU-accelerated embeddings object to convert your chunks into vectors and store them in ChromaDB.

from langchain_chroma import Chroma

# 1. Create a Vector Store from chunks using the GPU-accelerated embedding model
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings, # Uses the GPU instance created in Step 2
    persist_directory="./chroma_db/NK_CV_ver_7_pdf_full"
)

# 2. Define the Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector store created and retriever initialized.")

# LangChain Integration:

# OLD: from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM # This is the new, recommended class

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Initialize Ollama LLM
# This model's computation will be handled by Ollama, which uses your GPU.
llm = OllamaLLM(model="llama3.2:3b")
# to try: llama3.2:3b, qwen2.5:1.5b, tinyllama:1.1b
# ok: qwen3:0.6b, qwen2.5:1.5b; llama3.2:3b, tinyllama:1.1b
# !!! llama3.2:3b is better than both qwen's !!!
# !!! llama3.2:3b is better than tinyllama:1.1b  !!!
## --- failed: ## llama3:8b, qwen2.5:7b

# 2. Define Prompt Template
prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context. 
    Context: {context}
    Question: {input}
""")

# 3. Build the RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("LLM and RAG chain successfully configured.")

# Run the Query[ies] - Execute the final RAG chain.
question = "What are the main topics discussed in the document?"
response = retrieval_chain.invoke({"input": question})

print("\n--- RAG Response ---")
print(response["answer"])

# To Do:
# Make the Q&A's in a loop through a command prompt (for Q's) and print the answers
# ... until the User says /Bye!

question = "What can you tell us about Nedialko Krouchev's accomplisments at Plusgrade?"
response = retrieval_chain.invoke({"input": question})

print("\n--- RAG Response #3 ---")
print(response["answer"])

question = "Does the context contain anything about Nedialko Krouchev's accomplisments or job affiliations?"
response = retrieval_chain.invoke({"input": question})

print("\n--- RAG Response #4 ---")
print(response["answer"])