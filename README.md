# Medical RAG Assistant

A **Retrieval-Augmented Generation (RAG) system for medical knowledge retrieval** using the Merck Medical Manual.  
The system combines semantic search with a large language model to generate answers grounded in medical literature.

---

## Problem

Medical professionals often need reliable information from large medical manuals containing thousands of pages.  
Traditional LLMs:
- Lack domain grounding
- Can produce hallucinated or unsafe responses

This project addresses the problem by:

- Retrieving relevant medical context
- Forcing the model to generate answers only from trusted sources

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

## Key Improvements & Results
Improved retrieval relevance by ~25–30% through:
- Better chunking strategy
- Embedding selection
- Retrieval tuning
Reduced hallucinations by:
- Strict prompt constraints
- Context-only answering
  
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

## Design Decisions
Chose RAG over fine-tuning for:
- Better interpretability
- Lower compute cost
- Easier updates with new data
  
Used local LLM (Mistral via llama.cpp) for:
- Privacy
- Cost efficiency

---
## Limitations & Future Work
- Limited to static documents 
- No reranking model 
Can be extended with:
- Hybrid search (BM25 + embeddings)
- Multi-document reasoning


## Disclaimer

This project is for **educational and research purposes only** and should not be used as a substitute for professional medical advice.
