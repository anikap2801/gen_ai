import config
import re

def extract_level_number(level_str):
    """
    Extract numeric level from level strings like "L1", "L2 - Explanation", etc.
    Returns: int or None if unable to extract
    """
    if not level_str:
        return None
    
    # Try to find L[number] pattern
    match = re.search(r'L(\d)', str(level_str).upper())
    if match:
        return int(match.group(1))
    
    return None

def determine_gap(q_level_str, prep_level_str):
    """
    Explicitly compare cognitive levels using numeric extraction.
    
    Returns: (has_gap: bool, analysis: str)
    """
    q_level = extract_level_number(q_level_str)
    prep_level = extract_level_number(prep_level_str)
    
    if q_level is None or prep_level is None:
        return True, f"Unable to determine levels: Q={q_level_str}, Prep={prep_level_str}"
    
    # Clear logic: prep_level must be >= q_level
    if prep_level >= q_level:
        return False, f"Material covers sufficient depth (L{prep_level} >= L{q_level})"
    else:
        return True, f"Gap exists - Material at L{prep_level} but question requires L{q_level}"


def normalize_answer_text(answer_text):
    if not answer_text:
        return ""
    normalized = answer_text.lower()
    normalized = re.sub(r'[^a-z0-9 ]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def extract_predicted_answer(answer_analysis):
    patterns = [
        r'FINAL ANSWER:\s*(.+)',
        r'CORRECT ANSWER:\s*(.+)',
        r'ANSWER:\s*(.+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, answer_analysis, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def generate_gap_report(llm_client, question, q_level, prep_level, context, correct_answer=None):
    """
    Generate gap analysis comparing question cognitive level vs preparation level.
    
    Now includes:
    1. Rigorous answer reasoning using study material
    2. Explicit numeric level comparison
    3. Validation against dataset answer
    4. Optimized output without repetition
    """
    
    # ✅ IMPROVEMENT 1: Rigorous answer reasoning with clear option validation
    answer_reasoning_prompt = f"""
STEP-BY-STEP ANALYSIS - Verify each option.

QUESTION: {question}

STUDY MATERIAL (Reference):
{context}

FOR EACH OPTION, WRITE:
Option A: TRUE/FALSE - [Reason based on material]
Option B: TRUE/FALSE - [Reason based on material]
Option C: TRUE/FALSE - [Reason based on material]
Option D: TRUE/FALSE - [Reason based on material]

Then state the final answer as:
FINAL ANSWER: [A/B/C/D - option text]
"""

    answer_analysis = llm_client.generate(model=config.LLM_MODEL, prompt=answer_reasoning_prompt)
    predicted_answer = extract_predicted_answer(answer_analysis)
    
    # ✅ IMPROVEMENT 2: Validate prediction against dataset answer
    dataset_answer = correct_answer or "N/A"
    predicted_norm = normalize_answer_text(predicted_answer)
    dataset_norm = normalize_answer_text(dataset_answer)
    if predicted_norm and dataset_norm and (dataset_norm in predicted_norm or predicted_norm in dataset_norm):
        validation_note = "Prediction matches dataset answer."
    else:
        validation_note = f"Prediction differs from dataset answer (expected: {dataset_answer})."
    
    # ✅ IMPROVEMENT 3: Use explicit numeric level comparison
    has_gap, gap_analysis = determine_gap(q_level, prep_level)
    gap_status = "GAP EXISTS" if has_gap else "NO GAP"
    
    prompt = f"""
COGNITIVE GAP ANALYSIS

QUESTION: {question}

QUESTION LEVEL: {q_level}
MATERIAL LEVEL: {prep_level}

ANSWER ANALYSIS:
{answer_analysis}

DATASET ANSWER: {dataset_answer}
VALIDATION: {validation_note}

---
PROVIDE A BRIEF ASSESSMENT (3-4 sentences):
1. Is the answer supported by the material?
2. What is the gap status?
3. If there is a gap, why does it exist?
"""

    response = llm_client.generate(model=config.LLM_MODEL, prompt=prompt)
    return response.strip()