import pandas as pd
import openai
import os
import numpy as np
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def truncate_text(text, max_tokens=8000):
    """Truncate text to fit within token limit"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text


def get_embedding(text):
    try:
        # Truncate text before generating embedding
        truncated_text = truncate_text(text)
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=truncated_text,
            dimensions=256,
            encoding_format="float",
        )
        embedding_array = np.array(response.data[0].embedding)
        return embedding_array
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# Read and prepare data
df = pd.read_csv("F:/vecdb-compare/data/Resume/Resume.csv")
df = df.drop(columns=["Resume_html"])
df = df.drop(df.sample(n=1500, random_state=42).index)

# Check token lengths before processing
print("Checking token lengths...")
encoding = tiktoken.encoding_for_model("text-embedding-3-small")
df["token_count"] = df["Resume_str"].apply(lambda x: len(encoding.encode(x)))
print(f"Max tokens in Resume: {df['token_count'].max()}")
print(f"Average tokens in Resume: {df['token_count'].mean():.0f}")

# Generate embeddings
print("Generating embeddings...")
df["embedding"] = df["Resume_str"].apply(get_embedding)

# Save DataFrame with embeddings in pickle format
df.to_pickle("F:/vecdb-compare/data/processed/resumes_with_embeddings_256d.pkl")
print("Embeddings generated and saved!")

# Convert embeddings to list for CSV storage
df["embedding"] = df["embedding"].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

# Save as CSV
df.to_csv("F:/vecdb-compare/data/processed/resumes_with_embeddings_256d.csv", index=False)
print("Embeddings saved in both pickle and CSV formats!")
