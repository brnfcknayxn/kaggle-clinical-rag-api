import os
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 1. Initialize FastAPI
app = FastAPI(title="Clinical Trial AI Evaluator")

# 2. Load Databases & Models (Ye static hain, inko koi key nahi chahiye)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# 3. Define Tools & Schemas
@tool
def get_patient_data(patient_id: str) -> str:
    """Fetch patient medical records from SQL."""
    conn = sqlite3.connect('healthcare_data.db')
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
    record = cursor.fetchone()
    
    if record:
        data = dict(record)
        return json.dumps(data)
    return "Patient not found."

class PatientEvaluation(BaseModel):
    is_eligible: bool = Field(description="True ONLY if patient meets all criteria.")
    reason: str = Field(description="Explanation of why they passed or failed.")

class EvaluationRequest(BaseModel):
    patient_id: str

def get_best_rules(query: str):
    docs = vectordb.similarity_search(query, k=5)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    best_doc = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[0][1]
    return best_doc.page_content

# 4. API Route (Endpoint) - LLM is initialized dynamically here when called!
@app.post("/evaluate")
def evaluate_patient_endpoint(request: EvaluationRequest):
    try:
        # Ensure API key is picked up dynamically from environment
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Google API Key not found in environment variables.")

        # Initialize LLM right inside the request using the active API key
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=api_key)
        structured_llm = llm.with_structured_output(PatientEvaluation)

        prompt = ChatPromptTemplate.from_template("""
        SYSTEM: You are a strict medical AI evaluating clinical trial eligibility based on patient vitals and trial rules. 
        HALLUCINATION GUARDRAIL: ONLY use the clinical rules and SQL data provided below. Do not guess.

        CLINICAL RULES: {rules}
        PATIENT RECORD: {patient_data}
        """)

        sql_result = get_patient_data.invoke({"patient_id": request.patient_id})
        
        if "not found" in sql_result:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_data_dict = json.loads(sql_result)
        patient_disease = patient_data_dict.get('Disease', 'General Condition')
        
        best_rules = get_best_rules(f"What are the criteria for {patient_disease}?")
        
        chain = prompt | structured_llm
        result = chain.invoke({"rules": best_rules, "patient_data": sql_result})
        
        return {"status": "success", "patient_id": request.patient_id, "evaluation": result.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
