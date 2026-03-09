# Medical RAG Assistant

A **Retrieval-Augmented Generation (RAG) system for medical knowledge retrieval** using the Merck Medical Manual.  
The system combines semantic search with a large language model to generate answers grounded in medical literature.

---

## Problem

Medical professionals often need reliable information from large medical manuals containing thousands of pages.  
Standard language models may generate **hallucinated responses** when answering domain-specific questions.

This project addresses the problem using a **RAG pipeline** that retrieves relevant sections from the medical manual and generates answers based only on those sections.

---

## Approach

The system follows a typical RAG workflow:

```text
Medical Manual PDF
        ↓
Document Chunking
        ↓
Sentence Embeddings
        ↓
Vector Database (Chroma)
        ↓
Semantic Retrieval
        ↓
Mistral-7B LLM
        ↓
Grounded Answer
```

### Key Components

- **Document Processing:** PyMuPDF for loading the PDF  
- **Chunking:** Recursive text splitter with overlap  
- **Embeddings:** Sentence Transformers (GTE-Large)  
- **Vector Store:** Chroma database for similarity search  
- **LLM:** Mistral-7B Instruct via llama.cpp  

---

## Tech Stack

- Python  
- LangChain  
- Chroma Vector Database  
- Sentence Transformers  
- Mistral-7B (GGUF via llama.cpp)  
- PyMuPDF  

---

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/medical-rag-assistant.git
cd medical-rag-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the assistant:

```bash
python medicalrag.py
```

---

## Notes

- The vector database is automatically created on the first run.
- The system answers questions strictly based on retrieved document context.

---

## Disclaimer

This project is for **educational and research purposes only** and should not be used as a substitute for professional medical advice.
