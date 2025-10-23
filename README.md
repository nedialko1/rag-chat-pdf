# 
# **Local Retrieval-Augmented Generation (RAG) System**
This is a ... 

*Python-app for Screening Job-candidates Resumes in PDF 
*** Pocket Edition*

This system provides a high-performance, 
fully local RAG pipeline using **Ollama** for large language models 
(LLMs) and **Local choices for document embeddings** of a fully integrated 
`LocalEmbeddings(BaseModel, Embeddings)` class 
or alternatively a localhost-provided `FastAPI`-based Service, 
ensuring complete data privacy and control.

## I. **Prerequisites**

1. **Python Environment:** Python 3.9+ and a virtual environment (.venv) initialized.
1. **Ollama:** Ensure the Ollama app server is installed and running on your system.
1. **Model(s):** Pull the necessary LLM. 
.*Recommended:* the efficient `llama3.2:3b` model for low-VRAM/RAM systems:
.`ollama pull llama3.2:3b``
.You are also welcome to experiment with any LLM/Embedding Model instead - 
.provided that they fit inside your available resources
.Moreover, you may pull the needed additional models using the Ollama CLI:

**Dependencies:** Install all required Python packages:
`pip install -r requirements.txt`

## **II. Recommended Setup** - the Simplest way to get started
Just run the following commands (separately using the same *one* CLI,
or inside a Python IDE)

``` bash
# Ingest the PDF:
python ingest_pdf.py [--mode full]
# Start the RAG Application:
python interactive_rag_qa.py
```
**Notes:**

- Foremost, notice that the above python codes may be run without parameters -
and hence directly 'played' from the Python IDE
- The default PDF ingestion `mode` parameter is `chunked`;
the optional `full` value means that the whole PDF is vectorized
as one contiguous piece - this mode is only recommended for brief 
documents of for cases when important content may be well perceived
only when multiple pages are interpreted together

- Last but not least, Two- or Three-Terminal Workflows are also possible,
Such will do document ingesting and embedding as HTTP POST (Embedding),
or swagger-UX (PDF Ingestion) services,
and are discussed in the next section.

## II. **Separate Two- or Three-Terminal Workflows**

``` bash
# 1. LLM for Generation (Q&A)
ollama pull llama3.2:3b
# 2. Embedding Model (used by the FastAPI service)
ollama pull all-minilm
# **Install Dependencies**: Install the required Python packages:
pip install -r requirements.txt
``` 

## **Startup Workflow (Three Steps)**

Here, you'd need two, three separate terminal windows to run 
the separate components of the full application suite. 
This flow is not recommended for laptops and is provided 
more in terms of looking ahead...

| **Terminal** | **Service**                | **Command**                   | **Notes**                                                                                          |
|:-------------|:---------------------------|:------------------------------|:---------------------------------------------------------------------------------------------------|
| **0**        | **Embedding Service**      | `python embedding_service.py` | Starts the fast local embedding API (must be running first).                                       |
| **1**        | **Ingestion Service**      | `python ingestion_service.py` | Selects a PDF and creates the ChromaDB vector store. Using `--mode chunked` is highly recommended. |
| **2**        | **RAG-A Q&A Application** | `python interactive_rag_qa.py` | Interactive CLI console application to ask questions against the ingested document.                |


## **Performance Note:** 
The very first query will load the llama3.2:3b model and take the longest time (100-200 seconds). 
A subsequent identical query may be much faster (~15-20 seconds) upon an LL model's successful caching. 
Model swapped in/out of memory may potentially lead to long delays answering questions.

## III. **Key Architectural Insights (Optimized for low VRAM/RAM)**

This pipeline was specifically optimized for resource-constrained hardware, 
yielding the following general results and observations:

1. *Chunking* is key: The `--mode chunked` ingestion strategy is practically mandatory in most cases.
It makes the Synthesis (A) stage **~ 3 times faster** and drastically more reliable by providing 
the small LLM with a manageable, focused context, preventing memory thrashing. 
1. **Fully Local, High-Speed RAG:** Combining the local Embedding Service and 
local Ollama LLM, achieved a pipeline with *Synthesis times* **consistently in the seconds range**
on subsequent (warm) queries.
1. **Prompt Fidelity:** The most recent `interactive_rag_qa.py` version 
uses a more constraining prompt template to ensure that the modest-sized LLM still maintains 
the professional-looking structure and formatting, and that the RAG adheres strictly to the RAG rules
(e.g. politely refusing to answer when the context lacks information for any given question,
focusing on the PDF screening task at hande).

## IV. **Important Warning Note on Re-Ingestion (Prevent PDF/ChromaDB File Locking)**

**Do not attempt to re-ingest a document**, while the Q&A application is running on the same one!

When the `interactive_rag_qa.py` CLI app is running, it maintains a *read lock* 
on the related ChromaDB elements. Attempting to overwrite the database while it is in use 
will result in a "Critical File Lock Error (e.g. in the MS Windows OS, a WinError 32)".

Before re-ingesting a PDF, quit the Q&A application by typing `exit` or `quit`.

## **Colophon**

**Nedialko Krouchev (c) 2025** 

All rights reserved on original ideas and contributions.
Inspiration and baselines provided by *Google Gemini* (tm).

Special thanks to the creators of Ollama, pypdf, the langchain LLM integration tools suite, 
the Chroma DB, the Hugging-face sentence-transformers etc.

## **Future Ambitions (The Next Chapter):**

Importantly, this successful, highly optimized local pipeline 
was built for and developed for a 'pre-historic' and certainly pre-LLM laptop 
(released by Dell in late 2015 :) 
Now let's move on outside-the-box (literally :) and onto even more fascinating topics - 
such as optimizing the LLMs and the associated AI-agent pipelines. 
A specific ambition would be to blow down the Gargantuan sizes of the models' parameters 
and attack the resources-voracious modern Goliath.

This proof-of-concept and exploratory little project serves as the foundation 
for the next steps: 
Such as applying well-known optimization principles and strategies to the language models 
with the goal of gently reducing the necessary parameter sizes 
and resource requirements in general, toward the next generation of AI/ML models and agents.
