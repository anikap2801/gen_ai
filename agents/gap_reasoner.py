import config

def generate_gap_report(llm_client, question, q_level, prep_level, context, correct_answer):
    """
    Generate gap analysis comparing question cognitive level vs preparation level.
    
    Logic:
    - No Gap: If Prep_Level >= Q_Level → Study material is suitable
    - Gap Exists: If Prep_Level < Q_Level → Identify missing knowledge
    """
    prompt = f"""
COGNITIVE GAP ANALYSIS - Compare Question Requirements vs Study Material

QUESTION DETAILS:
{question}

VERIFIED CORRECT ANSWER (from assessment data): {correct_answer}
QUESTION COGNITIVE LEVEL (from assessment data): {q_level}
STUDY MATERIAL COGNITIVE LEVEL: {prep_level}

RETRIEVED STUDY MATERIAL:
{context}

GAP ANALYSIS FRAMEWORK:

1. ANSWER VERIFICATION: ✓ Confirmed from assessment dataset (not LLM-generated)

2. COGNITIVE LEVEL COMPARISON:
   - Required: {q_level}
   - Available: {prep_level}

3. GAP DETERMINATION:
   IF prep_level >= question_level → NO GAP (material is sufficient)
   IF prep_level < question_level → GAP EXISTS (material lacks required depth)

4. IF GAP EXISTS - IDENTIFY SPECIFIC DEFICIENCIES:
   - What cognitive skills are missing?
   - What specific knowledge gaps exist?
   - How can the study material be improved?

OUTPUT FORMAT:
## Cognitive Gap Assessment

**Answer Verification:** ✓ {correct_answer} (confirmed from dataset)
**Level Comparison:** Required {q_level} vs Available {prep_level}
**Gap Status:** [NO GAP / GAP EXISTS]

**Analysis:**
[If no gap: "Study material provides adequate cognitive depth for this question"]
[If gap exists: "Specific deficiencies and improvement recommendations"]
"""

    response = llm_client.generate(model=config.LLM_MODEL, prompt=prompt)
    return response.strip()