⚖️ Legal AI — RAG-Powered Legal Document Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that lets you query legal documents in natural language — built with a clean modular pipeline and a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-blueviolet?style=for-the-badge)
![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


# Overview :-
Legal AI is a Retrieval-Augmented Generation (RAG) application that enables users to ask natural language questions over a corpus of legal documents. Instead of relying on a generic LLM that may hallucinate legal facts, the system retrieves the most semantically relevant passages from real legal documents and grounds the LLM's response in that context.
The project is built with a fully modular pipeline — each stage (parsing, embedding, retrieval, generation) is its own independent module — and exposed through a clean Flask web interface.

# How it Works :-

```
Legal Documents (PDFs/Text)
        │
        ▼
   parser.py    ──→  Extract & chunk text from legal documents
        │
        ▼
embeddings.py   ──→  Convert chunks into vector embeddings & store in vector DB
        │
        ▼
 retrieval.py   ──→  On user query: semantic similarity search → top-k chunks
        │
        ▼
    llm.py      ──→  Feed retrieved context + query to LLM → grounded answer
        │
        ▼
   main.py      ──→  Flask web app: serves the UI, handles requests end-to-end
```

# Key Features :-

1) Natural language Q&A over legal documents — no keyword search required
2) RAG architecture — answers are grounded in retrieved document passages, reducing hallucinations
3) Modular pipeline — parser, embeddings, retrieval, and LLM are fully decoupled
4) Flask web interface — clean browser-based UI with HTML templates
5) Vector similarity search — semantically finds the most relevant legal clauses or sections
6) Modern Python tooling — uses uv for fast, reproducible dependency management

# Tech Stacks :-


| Component         | Technology                     |
|-------------------|--------------------------------|
| Web Framework     | Flask                          |
| LLM Integration   | OpenAI API / Local LLM         |
| Embeddings        | Sentence Transformers / OpenAI |
| Vector Store      | FAISS / ChromaDB               |
| Document Parsing  | PyMuPDF / pdfplumber           |
| Package Manager   | uv                             |
| Language          | Python 3.11+                   |
| Frontend          | HTML / Jinja2 Templates        |


# Project Structure :-

```
Legal-AI-Rag/
│
├── legal_ai/               # Core Python package
├── data/                   # Legal documents (PDFs, text files)
├── templates/              # Flask HTML templates (web UI)
│
├── parser.py               # Extracts and chunks text from legal documents
├── embeddings.py           # Generates and stores vector embeddings
├── retrieval.py            # Semantic search — finds relevant document chunks
├── llm.py                  # LLM integration — generates grounded answers
├── utils.py                # Shared utility functions
├── main.py                 # Flask app entry point
│
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency versions (uv)
└── .gitignore
```
# Getting Started Prerequisites :-

1) Python 3.11+
2) uv (recommended) or pip
3) An OpenAI API key (or a compatible local LLM)

# Installation :-
```bash
# Clone the repository
git clone https://github.com/Abdeali-Badri/Legal-AI-Rag.git
cd Legal-AI-Rag
```
# Using uv (recommended) :-
```bash
uv sync
```
# Using pip :-
```bash
pip install -r requirements.txt
```
# Configuration :-
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
# Add Your Legal Documents :-
1) Place your legal PDF or text files inside the data/ directory.
2) Run the App
3) ```bashpython main.py```
Then open your browser and navigate to```bash http://localhost:5000 ```

# Pipeline Deep Dive :-
1. ```parser.py\``` — Document Parsing
Loads legal documents from the data/ folder, extracts raw text, and splits it into overlapping chunks optimized for semantic search.
2. ```embeddings.py\``` — Vector Embeddings
Converts text chunks into high-dimensional embeddings using a sentence transformer or OpenAI's embedding API, then stores them in a vector database for fast retrieval.
3. ```retrieval.py\``` — Semantic Retrieval
At query time, embeds the user's question and performs a cosine similarity search against the stored vectors to retrieve the top-k most relevant legal passages.
4. ```llm.py\``` — Answer Generation
Constructs a prompt by combining the retrieved passages with the user's question, then sends it to the LLM. The model answers strictly based on the retrieved context, minimizing hallucinations.
5. ```main.py\``` — Web Application
Flask application that ties the entire pipeline together and serves the HTML interface where users can type questions and receive answers.

# Use Cases :-

1) Query contracts, NDAs, and legal agreements in plain English
2) Search and summarize case law or regulatory documents
3) Understand clauses in legal documents without a lawyer
4) Build a private knowledge base over any legal corpus


# Future Improvements :-

1) Support for multi-document cross-referencing
2) Chat history and conversation memory
3) Document upload directly from the web UI
4) Reranking layer for improved retrieval precision
5) Authentication and user session management
6) Deployable Docker container


## Author

**Abdeali Badri**
 [github.com/Abdeali-Badri](https://github.com/Abdeali-Badri)





