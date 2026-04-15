import config

def extract_question_level(question_data):
    """
    Extract the cognitive level directly from question data.
    The expected_cognitive_level is defined in the study material for this question.
    """
    return question_data.get("expected_cognitive_level", "L2")

def analyze_question(llm_client, question_text):
    """
    Analyze the question text only for unders tanding.
    This is supplementary - the main level comes from extract_question_level().
    """
    prompt = f"""
    Analyze the following database normalization question.
    
    L1: Recall (Definitions)
    L2: Understanding (Explaining concepts)
    L3: Application (Simple rule application)
    L4: Analysis (Complex multi-step problems)
    
    Question: {question_text}
    
    Return only the level (e.g., L1, L2, L3, or L4) and a one-sentence justification.
    """
    response = llm_client.generate(model=config.LLM_MODEL, prompt=prompt)
    return response.strip()