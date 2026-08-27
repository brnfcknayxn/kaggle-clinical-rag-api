import sqlite3
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup SQL Database and Load Kaggle CSV
conn = sqlite3.connect('healthcare_data.db')

print("Loading patient_data.csv...")
# Read the CSV downloaded from Kaggle
patients_df = pd.read_csv('patient_data.csv')

# Generate a unique 'patient_id' for each row (P-001, P-002, etc.)
patient_ids = ['P-' + str(i).zfill(3) for i in range(1, len(patients_df) + 1)]
patients_df.insert(0, 'patient_id', patient_ids) # Insert at the first column

# Save the updated dataframe to SQLite
patients_df.to_sql('patients', conn, if_exists='replace', index=False)
print(f"SQL Database created successfully with {len(patients_df)} records!")

# 2. Setup Vector Database (ChromaDB)
# (Keeping the mock clinical trial rules for RAG testing)
clinical_trial_text = """
TRIAL ID: ONC-2026-X
Title: Phase II Study of TargetX in Advanced Melanoma.
Inclusion Criteria: 
- Patients must have histologically confirmed advanced melanoma.
- Age must be greater than or equal to 18 years.
- Fasting blood glucose must be under 120 mg/dL.
Exclusion Criteria:
- Patients with a history of severe cardiovascular disease are excluded.
- Blood pressure exceeding 140/90 mmHg.
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.create_documents([clinical_trial_text])

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db", collection_name="clinical_trials")
print("VectorDB created successfully!")
