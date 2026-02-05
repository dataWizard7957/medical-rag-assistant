# Medical RAG Assistant
A Retrieval-Augmented Generation system for medical knowledge retrieval, combining LLaMA with a custom retriever and embeddings.

## Problem
Medical professionals need reliable answers from large document sets without hallucinations. The system provides grounded, context-aware responses.

## Approach
- LLaMA model for text generation
- Dynamic prompt templates and chain-of-thought prompting
- RAG hyperparameter tuning
- Iterative evaluation with LLM-as-a-judge to ensure accuracy

## Results
- ~25–30% improvement in response relevance
- Reduced hallucinations and improved domain-specific accuracy

## Tech Stack
Python, PyTorch, Hugging Face
