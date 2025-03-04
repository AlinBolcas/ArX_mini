import json
import numpy as np
from typing import List, Dict, Optional, Callable, Union
import sys
import os

# Navigate one level up from "textGen" to "arx_mini" and add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Utils  # Now you can import Utils properly

class TextChunk:
    """Represents a chunk of text with metadata."""
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata or {}


class TextSplitter:
    """Splits text into chunks based on length and overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(TextChunk(text=chunk, metadata={"chunk_index": len(chunks)}))
            i += self.chunk_size - self.chunk_overlap  # Ensure overlap

        return chunks


class SimpleVectorStore:
    """Basic vector store using NumPy for similarity search."""
    
    def __init__(self):
        self.embeddings = None
        self.chunks = []

    def add(self, texts: List[str], embedding_function: Callable[[str], np.ndarray]):
        """Generate embeddings and store them with associated chunks."""
        vectors = np.array([embedding_function(text).reshape(-1) for text in texts])  # Ensure flat shape
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack((self.embeddings, vectors))
        self.chunks.extend(texts)

    def search(self, query: str, embedding_function: Callable[[str], np.ndarray], k: int = 3) -> List[Dict]:
        """Perform nearest neighbor search based on cosine similarity."""
        if self.embeddings is None:
            return []

        query_vec = embedding_function(query).reshape(-1)  # Ensure (1536,)
        similarities = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(similarities)[-k:][::-1]  # Get top K results

        return [{"text": self.chunks[idx], "score": float(similarities[idx])} for idx in top_indices]


class RAG:
    """Retrieval-Augmented Generation (RAG) system using NumPy-based vector search."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.vector_store = SimpleVectorStore()

    def ingest_documents(self, documents: Union[str, List[str]], embedding_function: Callable[[str], np.ndarray]):
        """Process documents into chunks and store their embeddings."""
        if isinstance(documents, str):
            documents = [documents]

        chunks = []
        for doc in documents:
            chunks.extend(self.splitter.split_text(doc))

        self.vector_store.add([chunk.text for chunk in chunks], embedding_function)

    def query(self, query_text: str, embedding_function: Callable[[str], np.ndarray], top_k: int = 3) -> List[Dict]:
        """Perform retrieval for a query."""
        return self.vector_store.search(query_text, embedding_function, top_k)

    def retrieve_context(self, documents: Union[str, List[str]], query_text: str, embedding_function: Callable[[str], np.ndarray], top_k: int = 3) -> str:
        """
        Full pipeline: Processes documents, performs similarity search, and returns formatted context.
        This function eliminates the need to manually call each step.
        """
        self.ingest_documents(documents, embedding_function)
        top_chunks = self.query(query_text, embedding_function, top_k=top_k)
        return "\n\n".join([chunk["text"] for chunk in top_chunks])
        
if __name__ == "__main__":
    # Import OAI module dynamically via Utils
    oai = Utils.import_file("oai.py")

    if not oai:
        print("Error: OAI module not found. Ensure oai.py is in the project directory.")
        exit(1)

    oai_instance = oai.OAI(api_keys_path=None)  # Using default API key path

    def embedding_function(text: str) -> np.ndarray:
        """Use OAI to generate embeddings for given text."""
        return np.array(oai_instance.get_embeddings(text)).reshape(-1)  # Ensure (1536,)

    # Load and process the document
    about_alin = Utils.load_file("about_Alin.md")

    if about_alin:
        rag = RAG(chunk_size=1000, chunk_overlap=200)

        # Get retrieved context automatically
        retrieved_context = rag.retrieve_context(about_alin, "Who is Alin?", embedding_function, top_k=3)

        # Construct the final prompt
        final_prompt = f"Context:\n{retrieved_context}\n\nQuestion: Who is Alin?\nAnswer:"

        # Get response from AI
        response = oai_instance.chat_completion(final_prompt)
        print("\n=== AI Response ===\n")
        print(response)
    else:
        print("Error: about_Alin.md not found.")
