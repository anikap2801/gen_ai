import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# File Names
STUDY_MATERIAL_PATH = os.path.join(DATA_DIR, "study_material", "normalization_core.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions", "assessment_questions.json")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
CHUNKS_PKL_PATH = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")

# RAG Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Lightweight and fast for one-week sprint
K_RELEVANT_CHUNKS = 10  # Number of candidates to retrieve for reranking
TOP_K_AFTER_RERANK = 3  # Final number of chunks after reranking

# LLM Settings (Adjust based on if you use Ollama, OpenAI, etc.)
LLM_MODEL = "llama3.1"