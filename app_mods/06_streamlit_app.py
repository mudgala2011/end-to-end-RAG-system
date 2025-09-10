import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import sys

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.append(src_path)

# Import search functions from existing files
from _04_query_db import search_similar_resumes
from _05_hybrid_search import hybrid_search

# Initialize session state for API key
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None

# Streamlit UI
st.title("Candidate Search ...")

# API key input section
with st.sidebar:
    st.markdown("## OpenAI API Key")
    input_api_key = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/account/api-keys",
    )
    if input_api_key:
        st.session_state.openai_api_key = input_api_key
        os.environ["OPENAI_API_KEY"] = input_api_key

# Add database connection parameters
db_params = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
}


# Main content
if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    # Search type selector
    search_type = st.radio("Select Search Type", ["Semantic Search", "Hybrid Search"])

    # Query input
    query = st.text_input(
        "Desired Profile", placeholder="Enter the job profile you're looking for..."
    )

    # Number of results selector
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if query:
        with st.spinner("Searching..."):
            if search_type == "Semantic Search":
                results = search_similar_resumes(
                    query, top_k, min_similarity=0.5, db_params=db_params
                )
                # Update column order to match the data returned from search
                columns = ["Category", "ID", "Resume", "Similarity"]
            else:
                results = hybrid_search(query, top_k, db_params=db_params)
                # Update column order to match the data returned from hybrid search
                columns = ["ID", "Category", "Resume", "total_score", "vector_score", "text_score"]

            # Create DataFrame with appropriate columns
            results_df = pd.DataFrame(results, columns=columns)

            if not results_df.empty:
                if search_type == "Hybrid Search":
                    results_df = results_df.rename(columns={"total_score": "Similarity"})
                    # Ensure consistent column order for both search types
                    results_df = results_df[["ID", "Category", "Resume", "Similarity"]]
                else:
                    # Reorder columns to match display order
                    results_df = results_df[["ID", "Category", "Resume", "Similarity"]]
                    # Swap ID and Category columns to correct the order
                    results_df = results_df.rename(columns={"ID": "temp_id", "Category": "ID"})
                    results_df = results_df.rename(columns={"temp_id": "Category"})

                # Format the results
                results_df["Similarity"] = results_df["Similarity"].apply(lambda x: f"{x:.2%}")
                results_df["Resume"] = results_df["Resume"].apply(lambda x: f"{x[:200]}...")

                # Display results
                st.dataframe(
                    results_df,
                    column_config={
                        "ID": st.column_config.NumberColumn("ID", help="Resume ID"),
                        "Category": st.column_config.TextColumn("Category", help="Job Category"),
                        "Resume": st.column_config.TextColumn(
                            "Resume Preview", help="First 200 characters"
                        ),
                        "Similarity": st.column_config.TextColumn(
                            "Match Score", help="Similarity score"
                        ),
                    },
                    hide_index=True,
                )
            else:
                st.warning("No results found.")
