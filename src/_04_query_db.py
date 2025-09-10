import openai
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Add to both files at the top where db_params is used
db_params = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}


def get_query_embedding(query_text):
    """Convert query text to embedding vector"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query_text,
            dimensions=256,
            encoding_format="float",
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def search_similar_resumes(query_text, top_k=5, min_similarity=0.5, db_params=None):
    """
    Perform semantic search with improved scoring

    Args:
        query_text: Search query
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold (0-1)
    """
    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        return []

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    try:
        # Improved scoring query
        cur.execute(
            """
            WITH scored_resumes AS (
                SELECT 
                    id,
                    category,
                    resume_text,
                    CASE 
                        -- Strong match: similarity > 0.8
                        WHEN (1 - (embedding <=> %s::vector)) > 0.8 THEN 
                            (1 - (embedding <=> %s::vector)) * 1.2
                        -- Medium match: similarity between 0.6 and 0.8
                        WHEN (1 - (embedding <=> %s::vector)) > 0.6 THEN 
                            (1 - (embedding <=> %s::vector)) * 1.1
                        ELSE 
                            1 - (embedding <=> %s::vector)
                    END as similarity
                FROM resumes
                WHERE 1 - (embedding <=> %s::vector) >= %s
            )
            SELECT 
                id,
                category,
                resume_text,
                LEAST(similarity, 1.0) as similarity
            FROM scored_resumes
            ORDER BY similarity DESC
            LIMIT %s;
            """,
            (
                query_embedding.tolist(),
                query_embedding.tolist(),
                query_embedding.tolist(),
                query_embedding.tolist(),
                query_embedding.tolist(),
                query_embedding.tolist(),
                min_similarity,
                top_k,
            ),
        )

        results = cur.fetchall()
        return results

    finally:
        cur.close()
        conn.close()


def main():
    # Example queries
    queries = [
        "experienced digital media expert with experience in television",
        "finance controller with leadership experience",
        "Senior leadership experience in global technology companies",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        results = search_similar_resumes(query, top_k=3)
        for idx, (category, id, resume_text, similarity) in enumerate(results, 1):
            print(f"\nMatch #{idx} (Similarity: {similarity:.2f})")
            print(f"ID: {id}")
            print(f"Category: {category}")
            print(f"Resume Preview: {resume_text[:200]}...")
            print("-" * 30)


if __name__ == "__main__":
    main()
