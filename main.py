import config
import requests

from utils.file_loader import load_json
from agents.question_analyzer import analyze_question, extract_question_level
from agents.retriever_agent import retrieve_context
from agents.prep_analyzer import analyze_preparation
from agents.gap_reasoner import generate_gap_report


# ✅ Ollama LLM Client
class LLMClient:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"

    def generate(self, model, prompt):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code != 200:
                return f"Error: {response.text}"

            return response.json().get("response", "").strip()

        except Exception as e:
            return f"LLM Error: {str(e)}"


def format_mcq(question, options):
    """Formats question + options properly for LLM"""
    formatted = question + "\n\nOptions:\n"
    for i, opt in enumerate(options):
        formatted += f"{chr(65+i)}. {opt}\n"
    return formatted


def run_pipeline():
    client = LLMClient()

    # Load dataset
    questions = load_json(config.QUESTIONS_PATH)
    total_questions = len(questions)
    questions_per_page = 10

    print("🎯 Cognitive Gap Analysis System\n")

    while True:
        print(f"\nAvailable Questions (Total: {total_questions}):\n")

        # Pagination logic
        page = 0
        while True:
            start_idx = page * questions_per_page
            end_idx = min(start_idx + questions_per_page, total_questions)
            
            print(f"\n--- Page {page + 1} (Questions {start_idx + 1}-{end_idx}) ---")
            for i in range(start_idx, end_idx):
                q = questions[i]
                print(f"{i+1}. {q['question'][:80]}...")

            print(f"\nNavigation: [n]ext page, [p]revious page, or enter question number (1-{total_questions})")
            choice = input(">> ").strip().lower()

            if choice == 'n' and end_idx < total_questions:
                page += 1
                continue
            elif choice == 'p' and page > 0:
                page -= 1
                continue
            elif choice.isdigit() and 1 <= int(choice) <= total_questions:
                selected_idx = int(choice) - 1
                break
            else:
                print("Invalid choice. Try again.\n")
                continue

        selected_q = questions[selected_idx]

        # ✅ Format MCQ properly
        full_question = format_mcq(
            selected_q["question"],
            selected_q["options"]
        )

        print("\n" + "="*70)
        print("📌 SELECTED QUESTION")
        print("="*70)
        print(full_question)

        # Agent 1: Question Analyzer (Extract cognitive level from question data)
        q_level = extract_question_level(selected_q)
        print(f"\n🧠 Question Level: {q_level}\n")

        # Agent 2: Retriever
        retrieved_context = retrieve_context(selected_q)
        print("📚 Retrieved Study Material:")
        print("-"*70)
        print(retrieved_context)
        print("-"*70)

        # Agent 3: Preparation Analyzer
        prep_level = analyze_preparation(client, retrieved_context)
        print(f"\n📊 Material Cognitive Level: {prep_level}\n")

        # Agent 4: Gap Reasoner
        report = generate_gap_report(
            client,
            full_question,
            q_level,
            prep_level,
            retrieved_context
        )

        print("🧩 GAP ANALYSIS REPORT")
        print("="*70)
        print(report)
        print("="*70)

        continue_choice = input("\nAnalyze another question? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\nThank you for using Cognitive Gap Analysis System!")
            break


if __name__ == "__main__":
    run_pipeline()


if __name__ == "__main__":
    run_pipeline()