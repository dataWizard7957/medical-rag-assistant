"""
Medical RAG System using Merck Medical Manual
Single-file implementation cleaned from Jupyter Notebook.
"""

import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# ----------------------------------------------------
# Load LLM
# ----------------------------------------------------

def load_llm():

    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file = "mistral-7b-instruct-v0.2.Q6_K.gguf"

    model_path = hf_hub_download(
        repo_id=model_name,
        filename=model_file
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=20,
        n_batch=64
    )

    return llm


# ----------------------------------------------------
# Chunking
# ----------------------------------------------------

def create_text_splitter():

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=256,
        chunk_overlap=50
    )


# ----------------------------------------------------
# Load PDF
# ----------------------------------------------------

def load_pdf(filepath):

    loader = PyMuPDFLoader(filepath)
    documents = loader.load()

    return documents


# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------

def load_embedding_model():

    embedding_model = SentenceTransformerEmbeddings(
        model_name="thenlper/gte-large"
    )

    return embedding_model


# ----------------------------------------------------
# Vector Database
# ----------------------------------------------------

def create_vectorstore(chunks, embedding_model, persist_dir):

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=persist_dir
    )

    vectorstore.persist()

    return vectorstore


# ----------------------------------------------------
# Retriever
# ----------------------------------------------------

def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever


# ----------------------------------------------------
# RAG Response
# ----------------------------------------------------

def generate_rag_response(llm, retriever, question):

    system_prompt = """
You are a medical information expert.
Answer strictly using the provided medical manual.
If the answer is not present say "I don't know".
"""

    docs = retriever.get_relevant_documents(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
{system_prompt}

###Context
{context}

###Question
{question}
"""

    response = llm(
        prompt=prompt,
        max_tokens=300,
        temperature=0,
        top_p=0.95
    )

    return response["choices"][0]["text"].strip()


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main():

    pdf_path = "medical_diagnosis_manual.pdf"
    vector_dir = "medical_db"

    print("Loading LLM...")
    llm = load_llm()

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    splitter = create_text_splitter()

    print("Chunking documents...")
    chunks = splitter.split_documents(documents)

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    # Check if vector DB already exists
    if os.path.exists(vector_dir):
        print("Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=vector_dir,
            embedding_function=embedding_model
        )
    else:
        print("Creating vector database...")
        vectorstore = create_vectorstore(
            chunks,
            embedding_model,
            vector_dir
        )

    retriever = create_retriever(vectorstore)

    question = "What is the protocol for managing sepsis in a critical care unit?"

    answer = generate_rag_response(llm, retriever, question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
