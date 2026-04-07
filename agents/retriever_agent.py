import faiss
import pickle
import config
import numpy as np
from utils.embedding_model import load_embedding_model

def retrieve_context(question_data):
    """
    Retrieve context using semantic search with topic and correct answer filtering.
    This improves precision by considering the question topic and expected answer.
    """
    # Load index and metadata
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_PKL_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    model = load_embedding_model()
    
    # Create enhanced query that includes topic and correct answer for better retrieval
    question_text = question_data["question"]
    topic = question_data.get("topic", "")
    correct_answer = question_data.get("correct_answer", "")
    
    # Enhanced query combines question, topic, and correct answer for more precise retrieval
    enhanced_query = f"{question_text} {topic} {correct_answer}"
    
    query_vector = np.array(model.encode([enhanced_query])).astype("float32")
    
    # Search K-nearest neighbors
    distances, indices = index.search(query_vector, config.K_RELEVANT_CHUNKS)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i != -1]
    return "\n\n".join(retrieved_chunks)