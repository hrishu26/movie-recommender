import streamlit as st
import pandas as pd
import ast
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

MOVIES_URL = "https://raw.githubusercontent.com/akshay-patel28/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv"
CREDITS_URL = "https://raw.githubusercontent.com/akshay-patel28/TMDB-5000-Movie-Dataset/master/tmdb_5000_credits.csv"

MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"


def download_if_not_exists(url: str, filename: str) -> None:
    if not os.path.exists(filename):
        st.info(f"Downloading {filename} ...")
        df = pd.read_csv(url)
        df.to_csv(filename, index=False)


def safe_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        return []


def extract_names(lst, max_items=None):
    out = []
    for obj in lst:
        name = obj.get("name")
        if name:
            out.append(name)
        if max_items and len(out) >= max_items:
            break
    return out


@st.cache_resource
def load_data_and_vectors():
    download_if_not_exists(MOVIES_URL, MOVIES_FILE)
    download_if_not_exists(CREDITS_URL, CREDITS_FILE)

    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    df = movies.merge(credits, on="title")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
    df["overview"] = df["overview"].fillna("").astype(str)

    df["genres"] = df["genres"].apply(safe_eval).apply(lambda x: extract_names(x))
    df["keywords"] = df["keywords"].apply(safe_eval).apply(lambda x: extract_names(x))
    df["cast"] = df["cast"].apply(safe_eval).apply(lambda x: extract_names(x, max_items=3))
    df["crew"] = df["crew"].apply(safe_eval)

    def get_director(crew_list):
        for p in crew_list:
            if p.get("job") == "Director" and p.get("name"):
                return [p["name"]]
        return []

    df["director"] = df["crew"].apply(get_director)

    def clean(tokens):
        return [t.replace(" ", "") for t in tokens if isinstance(t, str) and t.strip()]

    for col in ["genres", "keywords", "cast", "director"]:
        df[col] = df[col].apply(clean)

    df["tags"] = (
        df["overview"].str.lower().str.split()
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["director"]
    )
    df["tags"] = df["tags"].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"])  # sparse

    df["title"] = df["title"].astype(str)
    title_to_idx = {t: i for i, t in enumerate(df["title"].values)}

    return df[["movie_id", "title"]], vectors, title_to_idx


movies_df, vectors, title_to_idx = load_data_and_vectors()


def recommend(title: str, top_n: int = 5):
    if title not in title_to_idx:
        return []
    idx = title_to_idx[title]
    sims = linear_kernel(vectors[idx], vectors).flatten()
    top_idx = sims.argsort()[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]
    return movies_df.iloc[top_idx]["title"].tolist()


st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System (TMDB 5000)")

selected = st.selectbox("Select a movie", movies_df["title"].values)
top_n = st.slider("Number of recommendations", 3, 15, 5)

if st.button("Recommend"):
    recs = recommend(selected, top_n=top_n)
    if not recs:
        st.warning("Recommendations unavailable.")
    else:
        for r in recs:
            st.write("✅", r)
