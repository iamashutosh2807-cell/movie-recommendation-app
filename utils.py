# =============================================================
# utils.py — Helper functions for Movie Recommendation System
# =============================================================
# Updated: Real TMDB poster fetching via TMDB API v3
# =============================================================

import pandas as pd
import numpy as np
import os
import requests

# ── Load API key from environment variable ────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "01039236253954282c424437a6a2185d")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER_PATH = os.path.join(BASE_DIR, "assets", "poster_placeholder.jpg")


def fetch_poster(movie_title: str) -> str:
    """
    Fetch the poster URL for a given movie from the TMDB API.

    How it works:
    1. Search TMDB for the movie by title using /search/movie endpoint.
    2. Take the first result's poster_path.
    3. Prepend TMDB image base URL to build the full poster URL.
    4. Return URL — Streamlit's st.image() displays URLs directly.

    Falls back to local placeholder if API key is missing or request fails.
    """
    if not TMDB_API_KEY:
        return PLACEHOLDER_PATH

    try:
        response = requests.get(
            TMDB_SEARCH_URL,
            params={
                "api_key": TMDB_API_KEY,
                "query": movie_title,
                "language": "en-US",
                "page": 1,
            },
            timeout=5,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if results and results[0].get("poster_path"):
            return TMDB_IMAGE_BASE + results[0]["poster_path"]

    except requests.exceptions.RequestException:
        pass

    return PLACEHOLDER_PATH


def recommend(movie_title: str, df: pd.DataFrame, similarity: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    match = df[df["movie_title"].str.lower() == movie_title.lower()]

    if match.empty:
        return pd.DataFrame(columns=["movie_title", "genres", "similarity"])

    movie_idx = match.index[0]
    scores = list(enumerate(similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1: top_n + 1]

    movie_indices = [i for i, score in scores]
    result = df.iloc[movie_indices][["movie_title", "genres"]].copy()
    result["similarity"] = [round(score * 100, 1) for _, score in scores]
    result = result.reset_index(drop=True)

    return result


def get_movie_list(df: pd.DataFrame) -> list:
    return sorted(df["movie_title"].tolist())


def get_movie_info(movie_title: str, df: pd.DataFrame) -> dict:
    match = df[df["movie_title"].str.lower() == movie_title.lower()]
    if match.empty:
        return {}

    row = match.iloc[0]
    return {
        "movie_title": row["movie_title"],
        "genres": row["genres"],
        "overview": row["overview"],
    }