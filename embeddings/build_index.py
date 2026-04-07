import faiss
import pickle
import numpy as np
import os
import sys

# Ensure the root directory is in the path so we can import utils and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from utils.chunking import chunk_text_by_topic
from utils.embedding_model import load_embedding_model

def build_index():
    # 1. Chunk the text
    print(f"Reading study material from: {config.STUDY_MATERIAL_PATH}")
    if not os.path.exists(config.STUDY_MATERIAL_PATH):
        print("Error: Study material file not found!")
        return

    chunks = chunk_text_by_topic(config.STUDY_MATERIAL_PATH)
    print(f"Generated {len(chunks)} knowledge chunks.")

    # 2. Generate Embeddings
    print("Loading embedding model and generating vectors...")
    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    embeddings_array = np.array(embeddings).astype('float32')
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_array = embeddings_array / norms

    # 3. Create and Save FAISS Index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalized vectors)
    index.add(embeddings_array)

    print(f"Saving FAISS index to: {config.FAISS_INDEX_PATH}")
    faiss.write_index(index, config.FAISS_INDEX_PATH)

    # 4. Save Metadata (the actual text chunks)
    print(f"Saving chunk metadata to: {config.CHUNKS_PKL_PATH}")
    with open(config.CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Indexing Complete!")

if __name__ == "__main__":
    build_index()