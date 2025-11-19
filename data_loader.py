from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF and split it into overlapping text chunks.
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks: list[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks into vectors using SentenceTransformers.
    Returns a list of Python lists.
    """
    if not texts:
        return []

    embeddings = embed_model.encode(texts)
    return [emb.tolist() for emb in embeddings]
