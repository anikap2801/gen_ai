import config

def analyze_preparation(llm_client, retrieved_context):
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
    Return format: "L[X] - [brief explanation of what cognitive skills are covered]"

    Example: "L2 - Explains normalization concepts and relationships between normal forms"
    """
    response = llm_client.generate(model=config.LLM_MODEL, prompt=prompt)
    return response.strip()