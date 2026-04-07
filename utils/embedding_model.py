import config
import os

def load_embedding_model():
    """Loads the sentence transformer model for creating embeddings."""
    # Set HF_HOME to avoid Windows path normalization issues
    hf_home = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

def load_reranker_model():
    """Loads the cross-encoder model for reranking."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')