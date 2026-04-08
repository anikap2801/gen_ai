import config
import re

def analyze_preparation(llm_client, retrieved_context):
    """
    Analyze the cognitive depth level of retrieved study material.
    
    Strategy:
    1. First, try to extract DEPTH levels directly from chunks (more reliable)
    2. If not found, use LLM analysis as fallback
    """
    
    # ✅ IMPROVEMENT: Extract DEPTH levels directly from chunks
    depth_pattern = r'DEPTH:\s*(L\d)'
    depth_matches = re.findall(depth_pattern, retrieved_context, re.IGNORECASE)
    
    if depth_matches:
        # Convert to numbers and find the maximum
        levels = []
        for match in depth_matches:
            try:
                # match is already "L1", "L2", etc.
                level_num = int(match[1])  # Extract digit from "L1", "L2", etc.
                levels.append(level_num)
            except (IndexError, ValueError):
                continue
        
        if levels:
            max_level = max(levels)
            return f"L{max_level} - Highest level found in retrieved material"
    
    # Fallback: Use LLM if explicit DEPTH markers not found
    prompt = f"""
    PREPARATION MATERIAL ANALYSIS - Focus on Study Content Depth

    Analyze the retrieved study material chunks to determine the cognitive depth level.
    Look specifically for:

    L1 (Recall/Knowledge): Definitions, basic concepts, terminology, simple facts
    L2 (Understanding): Explanations, examples, conceptual relationships, basic principles
    L3 (Application): Rules, procedures, problem-solving approaches, analytical methods

    Retrieved Study Material:
    {retrieved_context}

    TASK: Identify the HIGHEST cognitive level present in this material.
    Return format: "L[X] - [brief explanation]"
    """
    response = llm_client.generate(model=config.LLM_MODEL, prompt=prompt)
    return response.strip()