import pandas as pd
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))


register_adapter(np.ndarray, addapt_numpy_array)

# Load the pickle file
print("Loading pickle file...")
df = pd.read_pickle("F:/vecdb-compare/data/processed/resumes_with_embeddings_256d.pkl")

# Local Docker database connection parameters
db_params = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}

print("Connecting to database...")
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Create table and vector extension if they don't exist
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cur.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    category TEXT,
    resume_text TEXT,
    embedding vector(256)
);
""")

# Create the vector similarity search index
cur.execute("""
CREATE INDEX IF NOT EXISTS resume_embedding_idx 
ON resumes 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""")

# Insert data
print("Inserting data...")
for idx, row in df.iterrows():
    cur.execute(
        """
        INSERT INTO resumes (category, resume_text, embedding) 
        VALUES (%s, %s, %s)
        """,
        (
            row["Category"],
            row["Resume_str"],
            row["embedding"].tolist(),
        ),
    )
    if idx % 100 == 0:
        print(f"Processed {idx} rows...")
        conn.commit()

conn.commit()
cur.close()
conn.close()

print("Data upload complete!")
