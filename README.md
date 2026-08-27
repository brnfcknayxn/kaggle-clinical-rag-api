# Clinical Trial AI Evaluator (RAG + Function Calling)

## Project Overview
This enterprise-grade AI backend automates the process of matching patients to complex clinical trials. It replaces manual document checking with an automated **Retrieval-Augmented Generation (RAG)** pipeline. 

By combining **structured patient data** (SQL/Kaggle dataset) with **unstructured clinical protocols** (VectorDB), the system uses LLM reasoning to deterministically evaluate patient eligibility and output strict JSON responses for front-end integrations.

## System Architecture
1. **Data Ingestion:** Reads synthetic patient data from a Kaggle dataset (`.csv`) and builds a relational `SQLite` database.
2. **Vectorization:** Chunks and embeds clinical trial protocols using HuggingFace Transformers, storing them in `ChromaDB`.
3. **API Routing:** `FastAPI` exposes a POST endpoint to receive patient IDs.
4. **Function Calling:** LangChain tools dynamically query the SQL database for real-time patient vitals.
5. **Cross-Encoder Re-Ranking:** Improves RAG precision by re-ranking vector search results before sending them to the LLM.
6. **Guardrails & Generation:** Gemini AI evaluates the context against strict Pydantic schemas to prevent hallucinations and return deterministic JSON.

## Tech Stack
* **Framework:** FastAPI, Python
* **AI/Orchestration:** LangChain, Google Gemini 2.0 Flash
* **Vector & ML:** ChromaDB, HuggingFace, Sentence-Transformers (Cross-Encoder)
* **Data Management:** SQLite, Pandas, Pydantic

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/kaggle-clinical-rag-api.git](https://github.com/YourUsername/kaggle-clinical-rag-api.git)
   cd kaggle-clinical-rag-api
