# 🎬 Movie Recommendation System

A clean, minimal **Content-Based Movie Recommendation App** built with Python and Streamlit.

---

## 🧠 How It Works

This app uses **Content-Based Filtering** — it recommends movies similar to the one you select, using the movie's own metadata (genres, keywords, plot overview).

| Step | What happens |
|------|-------------|
| 1 | Load movie dataset (CSV with genres, keywords, overview) |
| 2 | Combine text features into a single `tags` string per movie |
| 3 | Apply **TF-IDF Vectorizer** to convert text → numeric vectors |
| 4 | Compute **Cosine Similarity** between all movie pairs |
| 5 | Return top-5 most similar movies for the selected title |

> **Why TF-IDF + Cosine Similarity?**
> It's simple, fast, interpretable, and perfect for text-based content recommendations without needing any user history.

---

## 📁 Project Structure

```
movie-recommender/
│
├── app.py               ← Streamlit UI (frontend)
├── model.py             ← ML pipeline (TF-IDF + Cosine Similarity)
├── utils.py             ← Helper functions (recommend, fetch_poster, etc.)
├── movies.csv           ← Sample dataset (60 movies)
├── requirements.txt     ← Python dependencies
├── setup.sh             ← One-command local setup script
├── .gitignore
├── README.md
└── assets/
    └── poster_placeholder.jpg
```

---

## 🚀 Running Locally

### Option A — Manual setup

```bash
# 1. Clone the project
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Option B — One-command setup (Linux/macOS)

```bash
bash setup.sh
```

---

## ☁️ Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the **easiest and free** way to deploy this app.

1. Push your project to a **public GitHub repository**.
2. Go to → [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub.
4. Click **"New app"**.
5. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
6. Click **Deploy** — your app will be live in ~2 minutes!

> 💡 Streamlit Cloud automatically reads `requirements.txt` and installs all packages.

---

## ❌ Why NOT Netlify?

> Netlify only supports **static frontend deployments** (HTML, CSS, JavaScript).
> This app runs a **Python backend** (Streamlit + scikit-learn), which Netlify cannot execute.
> Use **Streamlit Community Cloud** or **Render** or **Railway** instead.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF Vectorizer + Cosine Similarity |
| `numpy` | Array operations |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🎯 ML Techniques Used

- **TF-IDF (Term Frequency–Inverse Document Frequency)** — text vectorization
- **Cosine Similarity** — similarity measurement between movie vectors
- **Content-Based Filtering** — recommendation approach based on item attributes

---

## 📸 UI Preview

| Feature | Description |
|---------|-------------|
| Movie dropdown | Select from 60 pre-loaded movies |
| Movie info card | Shows genres and overview of selected movie |
| Recommend button | Triggers the ML pipeline |
| 5 recommendation cards | Displays top matches with similarity scores |
| How it works | Expandable section explaining the algorithm |

---

## 👤 Author

Built as a college AI/ML project — JIIT Noida  
Streamlit · scikit-learn · Python