# simple_rag.py

import os
from typing import List, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_and_chunk_pdf, embed_texts


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"
llm = genai.GenerativeModel(GEMINI_MODEL)
DOCUMENT_CHUNKS: List[str] = []
DOCUMENT_EMBEDDINGS: List[List[float]] = []
DOCUMENT_SOURCES: List[str] = []


def index_pdf_in_memory(path: str, source_name: str) -> int:
    """
    Load a PDF, chunk it, embed chunks, and store in in-memory lists.
    """
    global DOCUMENT_CHUNKS, DOCUMENT_EMBEDDINGS, DOCUMENT_SOURCES

    chunks = load_and_chunk_pdf(path)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)

    DOCUMENT_CHUNKS.extend(chunks)
    DOCUMENT_EMBEDDINGS.extend(embeddings)
    DOCUMENT_SOURCES.extend([source_name] * len(chunks))

    return len(chunks)


def retrieve_context_in_memory(query: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
    """
    Embed the query, compute cosine similarity against all stored embeddings,
    and return the top_k most similar chunks and their sources.
    """
    if not DOCUMENT_CHUNKS:
        return [], []

    query_emb = embed_texts([query])[0]
    
    import numpy as np

    query_vec = np.array(query_emb).reshape(1, -1)
    doc_matrix = np.array(DOCUMENT_EMBEDDINGS)  

    sims = cosine_similarity(query_vec, doc_matrix)[0]  

    top_indices = sims.argsort()[::-1][:top_k]

    contexts = [DOCUMENT_CHUNKS[i] for i in top_indices]
    sources = list({DOCUMENT_SOURCES[i] for i in top_indices})

    return contexts, sources


def answer_question_in_memory(query: str, top_k: int = 5) -> Tuple[str, List[str]]:
    """
    Retrieve relevant chunks and use Gemini to answer based on those chunks.
    """
    contexts, sources = retrieve_context_in_memory(query, top_k=top_k)
    if not contexts:
        return "I couldn't find relevant information in the PDFs indexed so far.", []

    context_str = "\n\n".join(contexts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.

Context:
{context_str}

Question: {query}

If the answer is not clearly in the context, say you don't know based on the provided documents.
"""

    resp = llm.generate_content(prompt)
    return resp.text, sources
