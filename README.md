# Cognitive Level Gap Analysis in Database Normalization

##  Project Overview
This project implements an **Agentic RAG (Retrieval-Augmented Generation)** pipeline to identify "Cognitive Gaps" between a student's study material and academic assessments. Focusing on **Database Normalization**, the system audits whether the depth of the curriculum (L1-L3) is sufficient to solve complex problems (L4) typically found in competitive exams like GATE.

##  The Agentic Workflow
Unlike standard RAG, our system utilizes four specialized agents to perform a multi-step reasoning chain:
1. **Question Analyzer**: Detects the required Bloom's Level (L1-L4) of the assessment question.
2. **Retriever Agent**: Performs semantic search using **FAISS** to find the most relevant study chunks.
3. **Prep Analyzer**: Evaluates the retrieved material's depth by identifying `DEPTH` tags and structural headers.
4. **Gap Reasoner**: Compares the preparation vs. the requirement to generate a detailed audit report and explanation of the gap.

##  Tech Stack
- **LLM**: Llama 3.1 (via Ollama)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Language**: Python 3.9+

##  Folder Structure
- `/data`: Contains the merged `normalization_core.txt` and `assessment_questions.json`.
- `/embeddings`: Stores the generated `.bin` index and chunk metadata.
- `/agents`: Modular Python scripts for each of the four agents.
- [cite_start]`/utils`: Helper scripts for chunking, embedding, and file loading [cite: 85-88].
