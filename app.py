# =============================================================
# app.py — Streamlit Frontend for Movie Recommendation System
# =============================================================
# Run with:  streamlit run app.py
# =============================================================

import streamlit as st
import os

# Local modules
from model import build_model
from utils import recommend, fetch_poster, get_movie_list, get_movie_info

# ── Page Configuration ────────────────────────────────────────
# Must be the FIRST Streamlit command in the script
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
# Light, clean styling that looks great on all screens
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #b0a8d6;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    /* ── Selected movie info card ── */
    .info-card {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        color: #e0e0f0;
    }
    .info-card h3 {
        font-family: 'Playfair Display', serif;
        color: #c8b8ff;
        margin-bottom: 0.3rem;
        font-size: 1.3rem;
    }
    .info-card .genre-tag {
        display: inline-block;
        background: #302b63;
        color: #c8b8ff;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px 3px 2px 0;
    }
    .info-card p.overview {
        font-size: 0.9rem;
        color: #a8a8c0;
        margin-top: 0.7rem;
        line-height: 1.6;
    }

    /* ── Recommendation card ── */
    .rec-card {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.1rem;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }
    .rec-card:hover {
        transform: translateY(-3px);
        border-color: #7b5ea7;
    }
    .rec-card .rank {
        font-size: 0.75rem;
        color: #7b5ea7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .rec-card h4 {
        font-family: 'Playfair Display', serif;
        color: #e8e0ff;
        font-size: 1.05rem;
        margin: 0.3rem 0 0.5rem;
    }
    .rec-card .genres {
        font-size: 0.78rem;
        color: #a8a8c0;
    }
    .rec-card .score-bar-bg {
        background: #2a2a4a;
        border-radius: 4px;
        height: 5px;
        margin-top: 0.8rem;
    }
    .rec-card .score-bar-fill {
        background: linear-gradient(90deg, #7b5ea7, #c8b8ff);
        border-radius: 4px;
        height: 5px;
    }
    .rec-card .score-label {
        font-size: 0.72rem;
        color: #7b5ea7;
        margin-top: 0.3rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #e8e0ff;
        border-left: 4px solid #7b5ea7;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── Streamlit overrides ── */
    .stSelectbox > div > div {
        background-color: #1e1e2e !important;
        border: 1px solid #3a3a5c !important;
        color: #e0e0f0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stMarkdownContainer"] { color: #e0e0f0; }
    .stButton > button {
        background: linear-gradient(135deg, #302b63, #7b5ea7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* ── Dark background ── */
    [data-testid="stAppViewContainer"] {
        background-color: #0f0e1a;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #4a4a6a;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load ML Model (cached so it runs only once) ───────────────
# st.cache_data prevents re-running the ML pipeline on every
# user interaction — a big performance win.
@st.cache_data
def load_model():
    return build_model()

df, similarity = load_model()

# ── Get movie list for dropdown ───────────────────────────────
movie_list = get_movie_list(df)

# ── Hero Section ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Select a movie and discover similar films powered by Content-Based Filtering & TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])

with col_left:
    selected_movie = st.selectbox(
        "🎥 Choose a movie you like:",
        options=movie_list,
        index=0,
    )

with col_right:
    st.markdown("<br>", unsafe_allow_html=True)   # align button vertically
    recommend_btn = st.button("✨ Recommend")

# ── Selected Movie Info ───────────────────────────────────────
info = get_movie_info(selected_movie, df)

if info:
    genre_tags = " ".join(
        [f'<span class="genre-tag">{g.strip()}</span>' for g in info["genres"].split()]
    )
    st.markdown(f"""
    <div class="info-card">
        <h3>📽️ {info['movie_title']}</h3>
        <div>{genre_tags}</div>
        <p class="overview">{info['overview']}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Recommendations ───────────────────────────────────────────
if recommend_btn:
    results = recommend(selected_movie, df, similarity, top_n=5)

    if results.empty:
        st.warning("⚠️ No recommendations found for this movie. Try another!")
    else:
        st.markdown('<p class="section-header">🍿 Top 5 Similar Movies</p>', unsafe_allow_html=True)

        # Render 5 cards in a single row
        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            sim_score = row["similarity"]
            fill_width = int(sim_score)    # percentage width for bar

            # Genre tags (just the first 2 to keep card compact)
            genre_parts = row["genres"].split()[:3]
            genres_display = " · ".join(genre_parts) if genre_parts else "N/A"

            with cols[i]:
                # Fetch poster — returns either a TMDB URL or local placeholder path
                poster = fetch_poster(row["movie_title"])

                # Display if it's a URL (TMDB) or a local file that exists
                if poster.startswith("http"):
                    st.image(poster, use_container_width=True)
                elif os.path.exists(poster):
                    st.image(poster, use_container_width=True)

                # Movie card HTML
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rank">#{i+1} Match</div>
                    <h4>{row['movie_title']}</h4>
                    <div class="genres">{genres_display}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{fill_width}%"></div>
                    </div>
                    <div class="score-label">Similarity: {sim_score}%</div>
                </div>
                """, unsafe_allow_html=True)

