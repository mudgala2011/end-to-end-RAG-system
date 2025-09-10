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


def hybrid_search(query_text, top_k=3, vector_weight=0.7, db_params=None):
    """
    Perform hybrid search using both vector similarity and text matching

    Args:
        query_text: Search query in natural language
        top_k: Number of results to return
        vector_weight: Weight for vector similarity (1-weight for text matching)
    """
    query_embedding = get_query_embedding(query_text)
    if query_embedding is None:
        return []

    db_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "password",
        "host": "localhost",
        "port": "5432",
    }

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    try:
        # Create text search index if it doesn't exist
        cur.execute("""
            CREATE INDEX IF NOT EXISTS resume_text_search_idx 
            ON resumes USING gin(to_tsvector('english', resume_text));
        """)

        # Hybrid search query
        cur.execute(
            """
            WITH scores AS (
                SELECT 
                    id,
                    category,
                    resume_text,
                    %s * (1 - (embedding <=> %s::vector)) as vector_score,
                    (1 - %s) * ts_rank_cd(
                        to_tsvector('english', resume_text),
                        plainto_tsquery('english', %s)
                    ) as text_score
                FROM resumes
                WHERE 
                    embedding <=> %s::vector < 0.8
                    OR to_tsvector('english', resume_text) @@ plainto_tsquery('english', %s)
            )
            SELECT 
                id,
                category,
                resume_text,
                vector_score + text_score as total_score,
                vector_score,
                text_score
            FROM scores
            ORDER BY total_score DESC
            LIMIT %s;
            """,
            (
                vector_weight,
                query_embedding.tolist(),
                vector_weight,
                query_text,
                query_embedding.tolist(),
                query_text,
                top_k,
            ),
        )

        results = cur.fetchall()
        return results

    finally:
        cur.close()
        conn.close()


def main():
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        print("\nSearching with hybrid approach (70% vector, 30% text)...")
        results = hybrid_search(query, top_k=5)

        for idx, (id, category, resume_text, total_score, vector_score, text_score) in enumerate(
            results, 1
        ):
            print(f"\nMatch #{idx}")
            print(f"ID: {id}")
            print(f"Category: {category}")
            print(f"Total Score: {total_score:.3f}")
            print(f"Vector Score: {vector_score:.3f}")
            print(f"Text Score: {text_score:.3f}")
            print(f"Resume Preview: {resume_text[:200]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
