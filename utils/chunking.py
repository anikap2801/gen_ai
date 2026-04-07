def chunk_text_by_topic(file_path):
    """
    Splits the normalization_core.txt into chunks based on the '### TOPIC' header.
    Each chunk represents one 'Knowledge Unit'.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by the topic header
    raw_chunks = content.split("### TOPIC:")
    chunks = []
    
    for rc in raw_chunks:
        if rc.strip():
            # Re-add the header label for context
            chunks.append(f"### TOPIC: {rc.strip()}")
            
    return chunks