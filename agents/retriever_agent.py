import faiss
import pickle
import config
import numpy as np
import re
from utils.embedding_model import load_embedding_model, load_reranker_model

def retrieve_context(question_data):
    """
    Improved retrieval using:
    1. Topic-based matching (exact keyword matching from question topic field)
    2. Semantic search with normalized query
    3. Cross-encoder reranking
    4. Deduplication to avoid redundant chunks
    
    This provides more accurate and relevant context for gap analysis.
    """
    # Load index and metadata
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_PKL_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    model = load_embedding_model()
    reranker = load_reranker_model()
    
    # Create query from question text and options for better retrieval
    question_text = question_data["question"]
    question_topic = question_data.get("topic", "").lower()  # Use topic field if available
    options = question_data.get("options", [])
    options_text = " ".join(options) if options else ""
    
    # Query combines question and answer options (prioritize question text)
    query = f"{question_text} {options_text}"
    
    # ✅ IMPROVEMENT 1: Topic-based exact matching
    topic_matched_indices = []
    if question_topic:
        for i, chunk in enumerate(chunks):
            # Check if question topic appears in chunk
            if question_topic in chunk.lower():
                topic_matched_indices.append(i)
    
    # Normalize query vector
    query_vector = np.array(model.encode([query])).astype("float32")
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # ✅ IMPROVEMENT 2: Semantic search with more candidates
    distances, indices = index.search(query_vector, min(config.K_RELEVANT_CHUNKS * 2, len(chunks)))
    
    # Merge topic-matched chunks with semantic search results
    candidate_indices = set(indices[0].tolist())
    candidate_indices.update(topic_matched_indices)
    candidate_indices.discard(-1)  # Remove invalid indices
    
    candidate_chunks = [chunks[i] for i in candidate_indices if i < len(chunks)]
    
    # ✅ IMPROVEMENT 3: Cross-encoder reranking for precision
    if candidate_chunks:
        pairs = [[query, chunk] for chunk in candidate_chunks]
        scores = reranker.predict(pairs)
        
        # Sort by reranker scores (higher is better)
        scored_chunks = list(zip(candidate_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # ✅ IMPROVEMENT 4: Deduplicate similar chunks
        unique_chunks = []
        seen_topics = set()
        
        for chunk, score in scored_chunks:
            # Extract topic name from chunk
            topic_match = re.search(r'### TOPIC: ([^#]+) ###', chunk)
            if topic_match:
                topic = topic_match.group(1).strip()
                if topic not in seen_topics:
                    unique_chunks.append(chunk)
                    seen_topics.add(topic)
            else:
                # If no topic found, add always (shouldn't happen with proper format)
                unique_chunks.append(chunk)
            
            if len(unique_chunks) >= config.TOP_K_AFTER_RERANK:
                break
        
        top_chunks = unique_chunks
    else:
        top_chunks = []
    
    # Return chunks with separator for clarity
    return "\n\n" + "="*80 + "\n".join([f"\n{chunk}" for chunk in top_chunks])