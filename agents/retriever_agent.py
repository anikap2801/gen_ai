import faiss
import pickle
import config
import numpy as np
from utils.embedding_model import load_embedding_model, load_reranker_model

def retrieve_context(question_data):
    """
    Retrieve context using semantic search with query normalization and reranking.
    This improves precision by using cosine similarity, querying question + options, and reranking top candidates.
    """
    # Load index and metadata
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_PKL_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    model = load_embedding_model()
    reranker = load_reranker_model()
    
    # Create query from question text and options for better retrieval
    question_text = question_data["question"]
    options = question_data.get("options", [])
    options_text = " ".join(options) if options else ""
    
    # Query combines question and answer options
    query = f"{question_text} {options_text}"
    
    # Normalize query vector
    query_vector = np.array(model.encode([query])).astype("float32")
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # Search for more candidates
    distances, indices = index.search(query_vector, config.K_RELEVANT_CHUNKS)
    
    # Get candidate chunks
    candidate_chunks = [chunks[i] for i in indices[0] if i != -1]
    
    # Rerank candidates using cross-encoder
    if candidate_chunks:
        pairs = [[query, chunk] for chunk in candidate_chunks]
        scores = reranker.predict(pairs)
        
        # Sort by reranker scores (higher is better)
        scored_chunks = list(zip(candidate_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top K after reranking
        top_chunks = [chunk for chunk, score in scored_chunks[:config.TOP_K_AFTER_RERANK]]
    else:
        top_chunks = []
    
    return "\n\n".join(top_chunks)