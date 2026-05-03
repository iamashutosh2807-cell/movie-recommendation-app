# =============================================================
# model.py — Core ML logic for Movie Recommendation System
# =============================================================
#
# WHY Content-Based Filtering + TF-IDF + Cosine Similarity?
# ----------------------------------------------------------
# - Content-Based Filtering is perfect when we have rich movie
#   metadata (genres, keywords, overview) but no user history.
# - TF-IDF (Term Frequency–Inverse Document Frequency) converts
#   text into numeric vectors, giving more weight to rare but
#   meaningful words and less weight to common words.
# - Cosine Similarity measures how similar two movies are by
#   comparing their TF-IDF vectors — angle between vectors,
#   not magnitude — which works great for text data.
#
# This combo is simple, fast, interpretable, and highly
# effective for a college-level recommendation project.
# =============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ── Path to dataset ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "movies.csv")


def load_data() -> pd.DataFrame:
    """
    Load and return the movies CSV as a DataFrame.
    """
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame:
    - Fill any missing text columns with empty strings so
      TF-IDF doesn't crash on NaN values.
    - Combine genres + keywords + overview into one 'tags'
      column that represents the movie's 'identity'.
    """
    # Fill NaN values in text columns
    for col in ["genres", "keywords", "overview"]:
        df[col] = df[col].fillna("")

    # Create a combined 'tags' column for feature extraction.
    # We repeat genres and keywords (by concatenating them twice)
    # to give them slightly more weight than the overview.
    df["tags"] = (
        df["genres"] + " " +
        df["keywords"] + " " +
        df["keywords"] + " " +   # keywords repeated for weight
        df["overview"]
    )

    # Convert to lowercase for consistency
    df["tags"] = df["tags"].str.lower()

    return df


def build_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build the TF-IDF matrix and compute pairwise cosine similarity.

    Steps:
    1. TfidfVectorizer converts each movie's 'tags' string into
       a vector of TF-IDF scores.
    2. cosine_similarity computes a (n x n) matrix where
       matrix[i][j] = similarity score between movie i and movie j.
    """
    # max_features=5000 keeps only top 5000 words — enough for
    # good accuracy while keeping memory usage low.
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

    # Fit and transform: builds the vocabulary and vectorises all movies
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    # Compute cosine similarity between every pair of movies
    similarity = cosine_similarity(tfidf_matrix)

    return similarity


def build_model():
    """
    Full pipeline: load → preprocess → build similarity matrix.
    Returns:
        df         — cleaned DataFrame
        similarity — (n x n) cosine similarity matrix
    """
    df = load_data()
    df = preprocess(df)
    similarity = build_similarity_matrix(df)
    return df, similarity