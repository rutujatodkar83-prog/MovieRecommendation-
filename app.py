"""
🎬 Movie Rating, Recommendation & Analysis Dashboard
Big Data Analytics – ISE Project

How the file is organised (top to bottom):
  1. Imports, page config, CSS, constants
  2. Helper functions (formatting, plotting)
  3. DATA LAYER   : load CSV -> detect columns -> clean -> build movie table
  4. ANALYSIS     : weighted rating, genre statistics, filters, insights
  5. RECOMMENDER  : TF-IDF + cosine similarity (with a simple fallback)
  6. PYSPARK      : optional distributed-style processing (real results only)
  7. UI SECTIONS  : one function per dashboard section
  8. main()       : puts everything together

Run with:  streamlit run app.py
"""

# ----------------------------------------------------------------------------
# 1. IMPORTS, PAGE CONFIG, CONSTANTS
# ----------------------------------------------------------------------------
import hashlib
import html
import importlib.util
import io
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import nlp_engine
import recommendation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Rating, Recommendation & Analysis",
                   page_icon="🎬", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "movies.csv"   # bundled synthetic demo data

RED, GOLD = "#E50914", "#F5C518"
COLORWAY = [RED, GOLD, "#FF7A59", "#8E7CC3", "#4FC3F7", "#66BB6A", "#F06292", "#FFB74D"]
SCALE = ["#2B1D3A", "#7A1F3D", RED, GOLD]            # continuous colour scale

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(229, 9, 20, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 30%, rgba(79, 195, 247, 0.12) 0%, transparent 40%),
                linear-gradient(180deg, #100b16 0%, #171120 100%);
    color: #e2e8f0;
}

section[data-testid="stSidebar"]{background:#0c0812;border-right:1px solid rgba(255,255,255,0.05);}
.block-container{padding-top:2rem;max-width:1400px;}

.hero {
    padding: 2.2rem 2.8rem;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 2.5rem;
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(16, 11, 22, 0.6) 100%);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}
.hero h1 { margin: 0; font-size: 3.2rem; line-height: 1.1; font-weight: 700; color: #ffffff; text-shadow: 0 2px 15px rgba(229,9,20,0.4); }
.hero .sub { color: #4FC3F7; font-weight: 600; letter-spacing: .15em; text-transform: uppercase; font-size: 0.95rem; margin-top: 0.8rem; }
.hero p { color: #cbd5e1; margin: 1.2rem 0 0 0; max-width: 950px; font-size: 1.15rem; line-height: 1.6; }

.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:1rem 0 2rem 0;}
.kpi {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.kpi:hover { border-color: rgba(79, 195, 247, 0.5); transform: translateY(-4px); background: rgba(255, 255, 255, 0.05); }
.kpi .label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 6px; }
.kpi .value { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1.1; text-shadow: 0 2px 5px rgba(0,0,0,0.4); }
.kpi .value.small { font-size: 1.2rem; }
.kpi .sub { color: #F5C518; font-size: 0.85rem; margin-top: 6px; }

.insight { background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05); border-left: 4px solid #F5C518; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; color: #e2e8f0; }
.vcard { background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 16px 20px; height: 100%; transition: all 0.2s ease; }
.vcard:hover { border-color: #66BB6A; background: rgba(0,0,0,0.5); }
.vcard b { color: #F5C518; }

.movie-card {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 100%;
    background: linear-gradient(90deg, rgba(229,9,20,0.08) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.movie-card:hover {
    border-color: rgba(229, 9, 20, 0.5);
    transform: translateY(-6px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.5), 0 0 20px rgba(229,9,20,0.25);
}
.movie-card:hover::before { opacity: 1; }
.mc-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; position: relative; z-index: 1; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
.mc-meta { font-size: 0.95rem; color: #94a3b8; margin-bottom: 16px; font-weight: 300; position: relative; z-index: 1; }
.mc-why { 
    font-size: 1rem; 
    color: #f1f5f9; 
    background: rgba(0, 0, 0, 0.5); 
    padding: 14px 18px; 
    border-radius: 10px; 
    border-left: 4px solid #E50914;
    position: relative; 
    z-index: 1; 
}

.stTextInput > div > div > input {
    font-size: 1.3rem;
    padding: 18px 22px;
    border-radius: 14px;
    border: 2px solid rgba(255,255,255,0.08);
    background: rgba(0,0,0,0.3) !important;
    color: white;
    transition: all 0.3s ease;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.2);
}
.stTextInput > div > div > input:focus {
    border-color: #E50914;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.2), 0 0 0 2px rgba(229,9,20,0.25);
}
</style>"""

# ----------------------------------------------------------------------------
# 2. SMALL HELPERS
# ----------------------------------------------------------------------------
PLOT_CONFIG = {"displaylogo": False,
               "toImageButtonOptions": {"format": "png", "scale": 2}}   # camera icon = PNG download


def esc(x) -> str:
    """HTML-escape any value (movie titles may contain & < > characters)."""
    return html.escape(str(x))


def fmt_int(x) -> str:
    return f"{int(x):,}" if pd.notna(x) else "—"


def show(fig):
    """Display a Plotly figure full-width (works on old and new Streamlit APIs)."""
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def show_df(df: pd.DataFrame, height=None):
    """Display a table full-width without the index column."""
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def style_fig(fig, title=None, height=420):
    """Apply the dark cinematic Plotly theme to every chart."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        height=height, margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#e8e8ee"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), colorway=COLORWAY)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def safe(title, fn, *args, **kwargs):
    """Run one dashboard section; if it fails show a warning instead of crashing the app."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:   # noqa: BLE001 - we deliberately catch everything here
        st.warning(f"⚠️ {title} could not be displayed: {exc}")
        return None


def kpi_cards(items):
    """items = [(icon, label, value, sub_text, small_font?)] -> responsive HTML grid."""
    cards = ""
    for icon, label, value, sub, small in items:
        cls = "value small" if small else "value"
        cards += (f"<div class='kpi'><div class='label'>{icon} {esc(label)}</div>"
                  f"<div class='{cls}'>{esc(value)}</div><div class='sub'>{esc(sub)}</div></div>")
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. DATA LAYER
# ----------------------------------------------------------------------------
def norm(name) -> str:
    """'Movie_ID ' -> 'movieid'  (lower-case, letters+digits only) for flexible matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Possible names for every field we care about. Order inside a list = priority.
ALIASES = {
    "movie_id": ["movieid", "movie_id", "itemid", "mid", "id"],
    "title": ["title", "movietitle", "moviename", "originaltitle", "name", "movie"],
    "genre": ["genres", "genre", "category", "categories"],
    "user_id": ["userid", "user", "reviewerid", "customerid", "reviewer"],
    "num_ratings": ["votecount", "numratings", "ratingcount", "ratingscount", "numvotes",
                    "votes", "nratings", "noofratings", "totalratings"],
    "rating": ["rating", "voteaverage", "averagerating", "avgrating", "imdbrating",
               "score", "stars", "meanrating"],
    "year": ["year", "releaseyear", "releasedate", "released", "startyear", "date"],
    "popularity": ["popularity", "popularityscore"],
    "timestamp": ["timestamp", "ratingtimestamp", "ratedat", "datetime", "reviewtime", "unixreviewtime"],
}
# Fallback: column name merely CONTAINS one of these words
KEYWORDS = {
    "movie_id": ["movieid", "itemid"], "title": ["title"], "genre": ["genre"],
    "user_id": ["userid"], "num_ratings": ["votecount", "ratingcount", "numvotes", "numratings"],
    "rating": ["rating", "score"], "year": ["year", "releasedate"], "popularity": ["popular"],
    "timestamp": ["timestamp"],
}
NUMERIC_FIELDS = {"rating", "num_ratings", "popularity"}
DETECT_ORDER = ["movie_id", "title", "genre", "user_id", "num_ratings", "rating",
                "year", "popularity", "timestamp"]
TEXT_HINTS = ["overview", "description", "plot", "keyword", "tag", "director", "cast",
              "actor", "tagline", "summary", "stars"]


def is_numeric_like(series: pd.Series) -> bool:
    sample = series.dropna().head(2000)
    return len(sample) > 0 and pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5


def detect_columns(df: pd.DataFrame) -> dict:
    """Guess which CSV column plays which role. Never assumes exact names."""
    cols = list(df.columns)
    normed = {c: norm(c) for c in cols}
    found, used = {}, set()

    def ok(field, col):
        if col in used:
            return False
        return is_numeric_like(df[col]) if field in NUMERIC_FIELDS else True

    for field in DETECT_ORDER:                      # pass 1: exact (normalised) name match
        for alias in ALIASES[field]:
            hit = next((c for c in cols if normed[c] == alias and ok(field, c)), None)
            if hit:
                found[field] = hit
                used.add(hit)
                break
    for field in DETECT_ORDER:                      # pass 2: name contains keyword
        if field in found:
            continue
        for kw in KEYWORDS[field]:
            hit = next((c for c in cols if kw in normed[c] and ok(field, c)), None)
            if hit:
                found[field] = hit
                used.add(hit)
                break
    return found


def detect_text_columns(df: pd.DataFrame, colmap: dict) -> list:
    """Columns usable for content-based recommendation (overview, keywords, cast ...)."""
    used = set(colmap.values())
    out = []
    for c in df.columns:
        n = norm(c)
        if c not in used and any(h in n for h in TEXT_HINTS) and df[c].dtype == object:
            out.append(c)
    return out[:4]


@st.cache_data(show_spinner="Reading CSV …")
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, trying common encodings and separators."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):        # semicolon separated file
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=";", low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            break
    return pd.DataFrame()


@st.cache_data(show_spinner="Preparing input files …")
def load_inputs(_main_bytes: bytes, _rating_bytes, sig: str):
    """
    Returns (dataframe, message).
    If a second 'ratings' file is given (MovieLens style: movies.csv + ratings.csv),
    both files are joined on the Movie ID column.
    """
    main = read_csv_bytes(_main_bytes)
    if _rating_bytes is None or main.empty:
        return main, None
    ratings = read_csv_bytes(_rating_bytes)
    if ratings.empty:
        return main, "The ratings file is empty – using only the main file."
    cm, cr = detect_columns(main), detect_columns(ratings)
    if "movie_id" not in cm or "movie_id" not in cr:
        return main, "Could not find a Movie ID column in both files, so they were not merged."
    ratings = ratings.rename(columns={cr["movie_id"]: cm["movie_id"]})
    extra = [c for c in main.columns if c == cm["movie_id"] or c not in ratings.columns]
    meta = main[extra].drop_duplicates(subset=cm["movie_id"])
    merged = ratings.merge(meta, on=cm["movie_id"], how="left")
    return merged, f"Merged ratings file ({len(ratings):,} rows) with movie file on '{cm['movie_id']}'."


# ---- genre / text parsing ------------------------------------------------------
def parse_genres(value) -> str:
    """
    'Action|Comedy', 'Action, Comedy' or "[{'id': 28, 'name': 'Action'}]" -> 'Action|Comedy'.
    Empty / '(no genres listed)' -> 'Unknown'.
    """
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    if "name" in value and "{" in value:                       # JSON-like (TMDB style)
        parts = re.findall(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", value)
    else:
        parts = re.split(r"[|,;/]", value)
    parts = [p.strip() for p in parts if p.strip() and "no genres" not in p.lower()]
    return "|".join(dict.fromkeys(parts)) if parts else "Unknown"


def clean_text(value) -> str:
    """Turn a text/JSON-like metadata cell into plain words for TF-IDF."""
    if not isinstance(value, str):
        return ""
    if "name" in value and "{" in value:
        names = re.findall(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", value)
        return " ".join(n.replace(" ", "") for n in names)
    return value.strip()


def parse_year(series: pd.Series) -> pd.Series:
    """Year numbers (1995) or full dates ('1995-12-15') -> float year, NaN if invalid."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.where((num >= 1800) & (num <= 2100)).astype(float)
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.year.astype(float)


def year_from_title(titles: pd.Series) -> pd.Series:
    """MovieLens titles look like 'Toy Story (1995)' -> 1995."""
    return pd.to_numeric(titles.astype(str).str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")


# ---- the main cleaning + aggregation step ----------------------------------------
@st.cache_data(show_spinner="Cleaning data and building the movie table …")
def prepare_data(_df: pd.DataFrame, sig: str, colmap_items: tuple, text_cols: tuple) -> dict:
    """
    1. remove duplicate rows, rows without rating / movie
    2. create internal columns (key, title, genres, year, rating, user ...)
    3. if the file holds one row per RATING  -> aggregate to one row per MOVIE
       (this is the classic group-by that Spark would do on big data)
    4. return the movie table + a cleaning report
    """
    colmap = dict(colmap_items)
    df = _df
    report = {"raw_rows": len(df), "raw_cols": df.shape[1],
              "missing_cells": int(df.isna().sum().sum()),
              "missing_pct": float(df.isna().sum().sum() / max(df.size, 1) * 100)}

    work = df.drop_duplicates()
    report["duplicates_removed"] = len(df) - len(work)

    rating = pd.to_numeric(work[colmap["rating"]], errors="coerce")
    key_raw = work[colmap["movie_id"]] if "movie_id" in colmap else work[colmap["title"]]
    valid = rating.notna() & key_raw.notna()
    report["invalid_rows_removed"] = int((~valid).sum())
    work, rating, key_raw = work[valid], rating[valid], key_raw[valid]
    if work.empty:
        raise ValueError("No valid rows remain after cleaning (rating/movie columns are empty or non-numeric).")

    key = key_raw.astype(str)
    title = work[colmap["title"]].astype(str).str.strip() if "title" in colmap else "Movie " + key
    title = title.where(title.ne("") & title.ne("nan"), "Movie " + key)

    genre_col = work[colmap["genre"]] if "genre" in colmap else pd.Series(np.nan, index=work.index)
    uniq = {v: parse_genres(v) for v in genre_col.dropna().unique()}      # parse unique values only (fast)
    genres = genre_col.map(uniq).fillna("Unknown")
    report["missing_genre_rows"] = int((genres == "Unknown").sum())

    if "year" in colmap:
        year = parse_year(work[colmap["year"]])
        if year.notna().mean() < 0.3:
            year = year.fillna(year_from_title(title))
    else:
        yt = year_from_title(title)
        year = yt if yt.notna().mean() > 0.3 else pd.Series(np.nan, index=work.index)

    base = pd.DataFrame({"key": key, "title": title, "genres": genres, "year": year, "rating": rating})
    if "user_id" in colmap:
        base["user"] = work[colmap["user_id"]].astype(str)
    if "popularity" in colmap:
        base["pop"] = pd.to_numeric(work[colmap["popularity"]], errors="coerce")
    if "num_ratings" in colmap:
        base["cnt"] = pd.to_numeric(work[colmap["num_ratings"]], errors="coerce")
    for i, c in enumerate(text_cols):
        base[f"txt{i}"] = work[c]

    # --- is each row ONE RATING (rating-level) or ONE MOVIE (movie-level)? ---
    rating_level = ("user" in base.columns) or bool(base["key"].duplicated().any())
    report["pairs_resolved"] = 0
    if rating_level and "user" in base.columns:
        before = len(base)
        base = base.drop_duplicates(subset=["user", "key"], keep="last")   # same user rated same movie twice
        report["pairs_resolved"] = before - len(base)

    txt_cols = [c for c in base.columns if c.startswith("txt")]
    if rating_level:
        agg = dict(title=("title", "first"), genres=("genres", "first"), year=("year", "first"),
                   avg_rating=("rating", "mean"), num_ratings=("rating", "size"))
        if "pop" in base.columns:
            agg["popularity"] = ("pop", "mean")
        for c in txt_cols:
            agg[c] = (c, "first")
        movies = base.groupby("key", sort=False).agg(**agg).reset_index()
        has_counts = True
    else:
        movies = base.rename(columns={"rating": "avg_rating"}).copy()
        has_counts = "cnt" in base.columns and base["cnt"].notna().any()
        movies["num_ratings"] = base["cnt"] if has_counts else np.nan
        if "pop" in base.columns:
            movies["popularity"] = base["pop"]
        movies = movies.drop(columns=[c for c in ["cnt", "pop", "user"] if c in movies.columns])

    pop_proxy = "popularity" not in movies.columns or movies["popularity"].isna().all()
    if pop_proxy:   # no popularity column -> use number of ratings as a proxy
        movies["popularity"] = movies["num_ratings"]

    # unique, readable labels (two different movies may share a title)
    dup = movies["title"].duplicated(keep=False) & movies["key"].ne(movies["title"])
    movies.loc[dup, "title"] = movies.loc[dup, "title"] + " [" + movies.loc[dup, "key"] + "]"
    if rating_level:
        base = base.merge(movies[["key", "title"]].rename(columns={"title": "label"}), on="key")
        base = base.drop(columns=["title"]).rename(columns={"label": "title"})

    # one text field for the recommender
    movies["text"] = (movies[txt_cols].apply(lambda col: col.map(clean_text)).agg(" ".join, axis=1)
                      if txt_cols else "")
    movies = movies.drop(columns=txt_cols)

    # Velocity information (only when a rating timestamp exists)
    report["velocity"] = None
    if rating_level and "timestamp" in colmap:
        try:
            raw_ts = work[colmap["timestamp"]]
            num = pd.to_numeric(raw_ts, errors="coerce")
            if num.notna().mean() > 0.5:
                unit = "ms" if num.median() > 1e11 else "s"
                ts = pd.to_datetime(num, unit=unit, errors="coerce")
            else:
                ts = pd.to_datetime(raw_ts, errors="coerce", format="mixed")
            ts = ts.dropna()
            days = max((ts.max() - ts.min()).days, 1)
            report["velocity"] = {"start": ts.min().date(), "end": ts.max().date(),
                                  "days": days, "per_day": len(ts) / days}
        except Exception:   # noqa: BLE001 - velocity is optional information only
            report["velocity"] = None

    meta = dict(
        rating_level=rating_level,
        has_genre=bool("genre" in colmap and (movies["genres"] != "Unknown").any()),
        has_year=bool(movies["year"].notna().any()),
        has_users="user" in base.columns and rating_level,
        has_counts=has_counts, pop_proxy=pop_proxy,
        has_text=bool(movies["text"].str.strip().ne("").any()),
        text_cols=list(text_cols), colmap=colmap, report=report)
    base = base[[c for c in ["key", "title", "genres", "rating", "user"] if c in base.columns]]
    return {"movies": movies, "base": base, "meta": meta}


# ----------------------------------------------------------------------------
# 4. ANALYSIS
# ----------------------------------------------------------------------------
def add_weighted_rating(movies: pd.DataFrame, has_counts: bool, pct: int):
    """
    IMDb-style weighted rating:  WR = v/(v+m) * R + m/(v+m) * C
      R = movie's average rating        v = number of ratings of the movie
      m = minimum ratings required      C = overall average rating
    m is taken as the `pct`-th percentile of all rating counts.
    """
    out = movies.copy()
    C = float(out["avg_rating"].mean())
    if not has_counts:
        out["weighted_rating"] = out["avg_rating"]
        return out, None, C
    m = float(out["num_ratings"].quantile(pct / 100))
    v, R = out["num_ratings"].fillna(0), out["avg_rating"]
    denom = v + m
    out["weighted_rating"] = np.where(denom > 0, v / denom * R + m / denom * C, C)
    return out, m, C


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """'Action|Comedy' (one cell) -> two rows, one per genre."""
    d = df.assign(genre=df["genres"].str.split("|")).explode("genre")
    d["genre"] = d["genre"].str.strip()
    return d[d["genre"].ne("")]


def genre_stats(df: pd.DataFrame, has_counts: bool) -> pd.DataFrame:
    """One row per genre: number of movies, average rating, total ratings, popularity."""
    g = explode_genres(df)
    g = g[g["genre"] != "Unknown"]
    if g.empty:
        return pd.DataFrame()
    if has_counts:
        g = g.assign(_w=g["avg_rating"] * g["num_ratings"])
        s = g.groupby("genre").agg(movies=("key", "nunique"), total_ratings=("num_ratings", "sum"),
                                   _w=("_w", "sum"), mean_movie_rating=("avg_rating", "mean"),
                                   popularity=("popularity", "sum"))
        # average of ALL ratings in the genre (movies with more ratings count more)
        s["avg_rating"] = np.where(s["total_ratings"] > 0, s["_w"] / s["total_ratings"], s["mean_movie_rating"])
        s = s.drop(columns="_w")
    else:
        s = g.groupby("genre").agg(movies=("key", "nunique"), avg_rating=("avg_rating", "mean"),
                                   popularity=("popularity", "sum"))
        s["total_ratings"] = np.nan
    return s.reset_index().sort_values("movies", ascending=False)


def best_genre(gs: pd.DataFrame):
    """Highest-rated genre, ignoring tiny genres (< 3 movies) when possible."""
    elig = gs[gs["movies"] >= 3]
    elig = gs if elig.empty else elig
    return elig.sort_values("avg_rating", ascending=False).iloc[0]


def apply_filters(df: pd.DataFrame, f: dict, meta: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f["genres"]:
        wanted = set(f["genres"])
        mask &= df["genres"].map(lambda g: bool(wanted.intersection(g.split("|"))))
    mask &= df["avg_rating"].between(*f["rating"])
    if meta["has_year"] and f["year_active"]:
        mask &= df["year"].between(*f["year"])
    if meta["has_counts"]:
        mask &= df["num_ratings"].fillna(0) >= f["min_ratings"]
    if f["search"].strip():
        mask &= df["title"].str.contains(re.escape(f["search"].strip()), case=False, na=False)
    return df[mask]


def build_insights(df, base_f, meta, gs) -> list:
    """Every sentence below is computed from the (filtered) data – nothing is hard-coded."""
    out = []
    has_counts = meta["has_counts"]

    if not gs.empty:
        g = best_genre(gs)
        out.append(f"🎭 <b>{esc(g['genre'])}</b> is the highest-rated genre with an average rating of "
                   f"<b>{g['avg_rating']:.2f}</b> across {int(g['movies'])} movies.")
    top = df.sort_values(["weighted_rating", "num_ratings"], ascending=False).iloc[0]
    out.append(f"🏆 <b>{esc(top['title'])}</b> has the highest weighted rating "
               f"(<b>{top['weighted_rating']:.2f}</b>; plain average {top['avg_rating']:.2f}"
               f"{'' if pd.isna(top['num_ratings']) else f', {int(top['num_ratings']):,} ratings'}).")
    if not gs.empty:
        if has_counts and gs["total_ratings"].notna().any():
            p = gs.sort_values("total_ratings", ascending=False).iloc[0]
            share = p["total_ratings"] / gs["total_ratings"].sum() * 100
            out.append(f"📝 <b>{esc(p['genre'])}</b> receives the most ratings: {int(p['total_ratings']):,} "
                       f"({share:.1f}% of all genre-rating pairs).")
        else:
            p = gs.iloc[0]
            out.append(f"📝 <b>{esc(p['genre'])}</b> is the most common genre with {int(p['movies'])} movies.")
    if has_counts and len(df) >= 10 and df["num_ratings"].nunique() > 1:
        rho = df["num_ratings"].corr(df["avg_rating"], method="spearman")
        if pd.notna(rho):
            strength = ("negligible" if abs(rho) < 0.1 else "weak" if abs(rho) < 0.3
                        else "moderate" if abs(rho) < 0.5 else "strong")
            direction = "positive" if rho > 0 else "negative"
            meaning = ("movies with more ratings tend to be rated higher" if rho >= 0.1 else
                       "movies with more ratings tend to be rated lower" if rho <= -0.1 else
                       "popularity and rating are essentially unrelated")
            out.append(f"📈 Popularity vs rating shows a <b>{strength} {direction}</b> correlation "
                       f"(Spearman ρ = {rho:.2f}): {meaning}.")
    if meta["has_year"]:
        yc = df["year"].dropna().astype(int).value_counts()
        if not yc.empty:
            out.append(f"📅 <b>{yc.index[0]}</b> had the most movie releases in this selection ({yc.iloc[0]} movies).")
        yr = df.dropna(subset=["year"]).groupby(df["year"].astype("Int64")).agg(n=("key", "size"), r=("avg_rating", "mean"))
        yr = yr[yr["n"] >= 5]
        if not yr.empty:
            out.append(f"⭐ Among years with at least 5 movies, <b>{int(yr['r'].idxmax())}</b> has the best average "
                       f"rating (<b>{yr['r'].max():.2f}</b>).")
    if has_counts and len(df) >= 20:
        hi, lo = df["avg_rating"].quantile(0.9), df["num_ratings"].median()
        gems = df[(df["avg_rating"] >= hi) & (df["num_ratings"] <= lo)].sort_values("avg_rating", ascending=False)
        if not gems.empty:
            out.append(f"💎 <b>{len(gems)}</b> movies are 'hidden gems' (top-10% average rating but fewer ratings than "
                       f"the median movie), e.g. <b>{esc(gems.iloc[0]['title'])}</b> ({gems.iloc[0]['avg_rating']:.2f}).")
    if meta["has_users"] and base_f is not None and len(base_f) > 0 and "user" in base_f.columns:
        counts = base_f.groupby("user").size().sort_values(ascending=False)
        k = max(1, int(len(counts) * 0.1))
        out.append(f"👥 The most active 10% of users ({k:,} users) produced "
                   f"<b>{counts.head(k).sum() / counts.sum() * 100:.0f}%</b> of all ratings – a typical long-tail pattern.")
    if base_f is not None and len(base_f) > 0:
        common = base_f["rating"].round(1).mode().iloc[0]
        out.append(f"🎯 The most frequently given rating value is <b>{common:g}</b>.")
    return out[:8]


# ----------------------------------------------------------------------------
# 5. RECOMMENDER  (TF-IDF + cosine similarity, with a simple fallback)
# ----------------------------------------------------------------------------
def make_documents(movies: pd.DataFrame) -> pd.Series:
    """One text 'document' per movie = genre tokens (+ overview/keywords/cast if available)."""
    def genre_tokens(g):
        toks = [re.sub(r"[^a-z0-9]", "", x.lower()) for x in g.split("|") if x != "Unknown"]
        return " ".join(f"genre_{t}" for t in toks if t)
    gt = movies["genres"].map(genre_tokens)
    has_text = movies["text"].str.strip().ne("").any()
    # repeat genre tokens so they are not drowned by long descriptions
    return (gt + " " + gt + " " + gt + " " + movies["text"]) if has_text else gt


@st.cache_data(show_spinner=False)
def build_tfidf(_docs: pd.Series, cache_key: str):
    """Fit TF-IDF once per dataset. Returns sparse matrix or None if there is nothing to learn from."""
    if _docs.str.strip().eq("").all():
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True)
        return vec.fit_transform(_docs)
    except ValueError:
        return None


def recommend(movies: pd.DataFrame, sel_key: str, allowed_keys: set, n: int, cache_key: str):
    """Return (recommendation table, explanation of the method used)."""
    movies = movies.reset_index(drop=True)
    pos = int(movies.index[movies["key"] == sel_key][0])
    matrix = build_tfidf(make_documents(movies), cache_key)

    if matrix is not None:
        sim = cosine_similarity(matrix[pos], matrix).ravel()
        has_text = movies["text"].str.strip().ne("").any()
        method = ("Content-based filtering: TF-IDF on genres + movie descriptions/metadata, ranked by cosine similarity."
                  if has_text else
                  "Content-based filtering: TF-IDF on genres, ranked by cosine similarity "
                  "(ties are broken by weighted rating).")
        min_sim = 1e-9
    else:                                   # fallback: similar rating level + similar popularity
        feats = pd.DataFrame({"r": movies["avg_rating"],
                              "p": np.log1p(movies["num_ratings"].fillna(movies["popularity"]).fillna(0))})
        feats = (feats - feats.min()) / (feats.max() - feats.min()).replace(0, 1)
        dist = np.sqrt(((feats - feats.iloc[pos]) ** 2).sum(axis=1)).values
        sim = 1 / (1 + dist)
        method = ("Fallback (no genre/text data): movies with the closest average rating and popularity "
                  "(similarity = 1 / (1 + distance)).")
        min_sim = 0.0

    res = movies.assign(similarity=np.round(sim, 4))
    res = res[(res["key"] != sel_key) & res["key"].isin(allowed_keys) & (res["similarity"] > min_sim)]
    res = res.sort_values(["similarity", "weighted_rating"], ascending=False).head(n)
    table = pd.DataFrame({"Movie": res["title"], "Genre": res["genres"].str.replace("|", ", ", regex=False),
                          "Rating": res["avg_rating"].round(2), "Similarity Score": res["similarity"]})
    return table.reset_index(drop=True), method


# ----------------------------------------------------------------------------
# 6. PYSPARK (optional) – only real results are displayed
# ----------------------------------------------------------------------------
def spark_status() -> tuple:
    """(is_pyspark_installed, is_java_available)"""
    return importlib.util.find_spec("pyspark") is not None, shutil.which("java") is not None


@st.cache_resource(show_spinner="Starting local Spark session …")
def get_spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[*]").appName("MovieDashboard")
            .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def run_pyspark_analysis(base: pd.DataFrame, rating_level: bool, pct: int, top_n: int) -> dict:
    """
    Re-does the key calculations with PySpark on the cleaned data:
    read -> clean -> group-by (movie) -> weighted rating -> group-by (genre).
    """
    from pyspark.sql import functions as F
    spark = get_spark()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "movies_clean.csv")
        base[[c for c in ["key", "title", "genres", "rating", "user"] if c in base.columns]].to_csv(path, index=False)
        t0 = time.time()
        sdf = (spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(path)
               .withColumn("rating", F.col("rating").cast("double")))
        n_raw = sdf.count()                                                     # 1. records processed
        # 2. preprocessing: drop missing values; a user may rate a movie only once, so
        #    duplicates are detected on (user, movie) - never on the movie alone.
        clean = sdf.dropna(subset=["key", "rating"])
        if "user" in sdf.columns:
            clean = clean.dropDuplicates(["user", "key"])
        n_clean = clean.count()
        per_movie = clean.groupBy("key").agg(                                   # 3. group-by movie
            F.first("title").alias("title"), F.count("rating").alias("num_ratings"),
            F.avg("rating").alias("avg_rating"))
        overall = per_movie.agg(F.avg("avg_rating")).first()[0]                 # C
        result = {"n_raw": n_raw, "n_clean": n_clean, "n_movies": per_movie.count(), "C": overall}
        if rating_level:                                                        # 4. weighted rating
            m = per_movie.approxQuantile("num_ratings", [pct / 100], 0.001)[0]
            v = F.col("num_ratings")
            wr = (v / (v + F.lit(m))) * F.col("avg_rating") + (F.lit(m) / (v + F.lit(m))) * F.lit(overall)
            result["m"] = m
            result["top"] = (per_movie.withColumn("weighted_rating", wr)
                             .orderBy(F.desc("weighted_rating")).limit(top_n).toPandas())
        genre = (clean.withColumn("genre", F.explode(F.split("genres", r"\|")))   # 5. group-by genre
                 .filter(F.col("genre") != "Unknown")
                 .groupBy("genre").agg(F.count("rating").alias("ratings"),
                                       F.countDistinct("key").alias("movies"),
                                       F.avg("rating").alias("avg_rating"))
                 .orderBy(F.desc("ratings")))
        result["genre"] = genre.toPandas()
        result["seconds"] = time.time() - t0
    return result


# ----------------------------------------------------------------------------
# 7. UI SECTIONS
# ----------------------------------------------------------------------------
def render_header():
    st.markdown(
        "<div class='hero'><h1>🎬 MovieMate</h1>"
        "<div class='sub'>Natural Language Movie Recommendation & Big Data Analytics</div>"
        "<p>Discover movies using natural-language preferences and explore insights from large-scale movie rating data.</p></div>", unsafe_allow_html=True)
def sidebar_inputs():
    """Upload widgets. Returns (main_bytes, ratings_bytes, label, using_sample)."""
    st.sidebar.title("🎬 Data source")
    up_main = st.sidebar.file_uploader("Upload movie dataset (CSV)", type=["csv"], key="up_main")
    up_rate = st.sidebar.file_uploader(
        "Optional: ratings CSV", type=["csv"], key="up_rate",
        help="For MovieLens-style data: upload movies.csv above and ratings.csv here. "
             "They are joined on the movie ID column.")
    if up_main is not None:
        return up_main.getvalue(), (up_rate.getvalue() if up_rate else None), up_main.name, False
    if up_rate is not None:
        return up_rate.getvalue(), None, up_rate.name, False
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes(), None, "movies.csv (bundled SYNTHETIC demo data)", True
    return None, None, "", False


def column_mapping_ui(df: pd.DataFrame, detected: dict, sig: str) -> dict:
    """Show what was auto-detected and let the user correct it."""
    labels = {"movie_id": "Movie ID", "title": "Movie title", "genre": "Genre", "rating": "Rating",
              "num_ratings": "Number of ratings", "user_id": "User ID", "year": "Year / release date",
              "popularity": "Popularity", "timestamp": "Rating timestamp"}
    options = ["(none)"] + list(df.columns)
    mapping = {}
    with st.sidebar.expander("🧭 Column mapping (auto-detected)"):
        st.caption("Fix any wrong guess here. Analyses that need a missing column are switched off automatically.")
        for field, label in labels.items():
            default = detected.get(field)
            idx = options.index(default) if default in options else 0
            choice = st.selectbox(label, options, index=idx, key=f"map_{field}_{sig}")
            if choice != "(none)":
                mapping[field] = choice
    return mapping


def reset_filters():
    """Callback of the Reset button: delete all stored filter widget values."""
    for k in [k for k in st.session_state.keys() if str(k).startswith("f_")]:
        del st.session_state[k]


def sidebar_filters(movies: pd.DataFrame, meta: dict, sig: str) -> dict:
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Filters")
    st.sidebar.button("🔄 Reset Filters", on_click=reset_filters, key="reset_btn")
    f = {}

    genre_list = sorted({g for gs in movies["genres"].unique() for g in gs.split("|") if g != "Unknown"})
    if genre_list:
        f["genres"] = st.sidebar.multiselect("Genre", genre_list, key=f"f_genres_{sig}", placeholder="All genres")
    else:
        f["genres"] = []
        st.sidebar.caption("Genre filter unavailable (no genre information).")

    lo, hi = float(np.floor(movies["avg_rating"].min() * 10) / 10), float(np.ceil(movies["avg_rating"].max() * 10) / 10)
    f["rating"] = (lo, hi)
    if hi > lo:
        f["rating"] = st.sidebar.slider("Rating range", lo, hi, (lo, hi), step=0.1, key=f"f_rating_{sig}")

    f["year"], f["year_active"] = (0, 0), False
    if meta["has_year"]:
        ylo, yhi = int(movies["year"].min()), int(movies["year"].max())
        if yhi > ylo:
            f["year"] = st.sidebar.slider("Movie year range", ylo, yhi, (ylo, yhi), key=f"f_year_{sig}")
            f["year_active"] = f["year"] != (ylo, yhi)
    else:
        st.sidebar.caption("Year filter unavailable (no year information).")

    f["min_ratings"] = 0
    max_n = int(movies["num_ratings"].max()) if meta["has_counts"] else 0
    if max_n > 0:
        f["min_ratings"] = st.sidebar.slider("Minimum number of ratings", 0, max_n, 0, key=f"f_minr_{sig}")
    else:
        st.sidebar.caption("Minimum-ratings filter unavailable (no rating counts).")

    f["search"] = st.sidebar.text_input("🔍 Search movie title", key=f"f_search_{sig}")
    f["top_n"] = st.sidebar.select_slider("Top N movies", options=[5, 10, 15, 20, 25, 30], value=10, key=f"f_topn_{sig}")
    f["pct"] = 80
    if meta["has_counts"]:
        with st.sidebar.expander("⚙️ Weighted-rating setting"):
            f["pct"] = st.slider("m = percentile of rating counts", 50, 99, 80, key=f"f_pct_{sig}",
                                 help="A movie needs about m ratings before its own average counts more than "
                                      "the overall average.")
    return f


def show_dataset_overview(raw: pd.DataFrame, report=None):
    st.subheader("📂 Dataset Preview & Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    st.markdown("**First 20 rows**")
    show_df(raw.head(20))
    left, right = st.columns(2)
    with left:
        st.markdown("**Columns, data types and missing values**")
        info = pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                             "Missing": raw.isna().sum().values,
                             "Missing %": (raw.isna().mean().values * 100).round(2),
                             "Unique values": raw.nunique().values})
        show_df(info, height=300)
    with right:
        st.markdown("**Numeric summary**")
        num = raw.select_dtypes(include="number")
        if num.empty:
            st.info("No numeric columns found.")
        else:
            show_df(num.describe().T.round(3).reset_index().rename(columns={"index": "Column"}), height=300)
    if report:
        st.markdown("**Cleaning report**")
        st.write(f"- Duplicate rows removed: **{report['duplicates_removed']:,}**\n"
                 f"- Rows without a valid rating / movie removed: **{report['invalid_rows_removed']:,}**\n"
                 f"- Repeated (user, movie) ratings resolved: **{report['pairs_resolved']:,}**\n"
                 f"- Rows with missing genre (kept as 'Unknown'): **{report['missing_genre_rows']:,}**")


def section_ratings(df, base_f, meta, gs):
    st.subheader("⭐ Movie Rating Analysis")
    show_raw = meta["rating_level"] and base_f is not None and len(base_f) > 0
    cols = st.columns(2) if show_raw else [st.container()]
    with cols[0]:                                                 # Chart 1a
        fig = px.histogram(df, x="avg_rating", nbins=30, color_discrete_sequence=[RED])
        fig.update_layout(bargap=0.05, xaxis_title="Average rating per movie", yaxis_title="Number of movies")
        fig.add_vline(x=df["avg_rating"].mean(), line_dash="dash", line_color=GOLD,
                      annotation_text=f"mean {df['avg_rating'].mean():.2f}")
        show(style_fig(fig, "Chart 1 – Distribution of movie average ratings", 380))
    if show_raw:                                                  # Chart 1b
        with cols[1]:
            vc = base_f["rating"].round(1).value_counts().sort_index().reset_index()
            vc.columns = ["rating", "count"]
            fig = px.bar(vc, x="rating", y="count", color_discrete_sequence=[GOLD])
            fig.update_layout(xaxis_title="Rating value given by users", yaxis_title="Number of ratings")
            show(style_fig(fig, "Chart 1b – How users rate (all individual ratings)", 380))

    c1, c2 = st.columns(2)
    if gs.empty:
        st.info("ℹ️ Genre charts are hidden because no genre information is available.")
    else:
        top_g = gs.sort_values("movies", ascending=False).head(30)
        with c1:                                                  # Chart 2
            d = top_g.sort_values("avg_rating", ascending=False)
            fig = px.bar(d, x="genre", y="avg_rating", color="avg_rating", color_continuous_scale=SCALE,
                         text=d["avg_rating"].round(2), hover_data={"movies": True})
            fig.update_yaxes(range=[max(0, d["avg_rating"].min() - 0.3), d["avg_rating"].max() + 0.1])
            fig.update_layout(xaxis_title="", yaxis_title="Average rating", coloraxis_showscale=False)
            show(style_fig(fig, "Chart 2 – Average rating by genre", 380))
            st.caption("Y-axis is zoomed to make small differences visible.")
        with c2:                                                  # Chart 3
            if meta["has_counts"] and top_g["total_ratings"].notna().any():
                d = top_g.sort_values("total_ratings", ascending=False)
                fig = px.bar(d, x="genre", y="total_ratings", color="total_ratings", color_continuous_scale=SCALE)
                fig.update_layout(xaxis_title="", yaxis_title="Total ratings", coloraxis_showscale=False)
                show(style_fig(fig, "Chart 3 – Number of ratings by genre", 380))
            else:
                st.info("ℹ️ Chart 3 is hidden: the dataset has no 'number of ratings' information.")

    if meta["has_counts"] and df["num_ratings"].notna().any():    # Chart 4
        d = df.dropna(subset=["num_ratings"]).copy()
        if len(d) > 5000:
            d = d.sample(5000, random_state=42)
            st.caption("Scatter plot shows a random sample of 5,000 movies for speed.")
        d["year_txt"] = d["year"].map(lambda y: "—" if pd.isna(y) else str(int(y)))
        if meta["has_genre"]:
            primary = d["genres"].str.split("|").str[0]
            top8 = df["genres"].str.split("|").str[0].value_counts().head(8).index
            d["Primary genre"] = primary.where(primary.isin(top8), "Other")
        else:
            d["Primary genre"] = "All movies"
        log_x = st.checkbox("Logarithmic x-axis (recommended when a few movies have very many ratings)", value=True)
        fig = px.scatter(d, x="num_ratings", y="avg_rating", color="Primary genre", log_x=log_x, opacity=0.8,
                         custom_data=["title", "genres", "year_txt"])
        fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Rating: %{y:.2f}<br>Number of ratings: %{x:,}"
                                        "<br>Genre: %{customdata[1]}<br>Year: %{customdata[2]}<extra></extra>")
        fig.update_layout(xaxis_title="Number of ratings", yaxis_title="Average rating")
        show(style_fig(fig, "Chart 4 – Rating vs number of ratings", 480))
        if len(d) >= 10 and d["num_ratings"].nunique() > 1:
            st.caption(f"Spearman correlation between number of ratings and average rating: "
                       f"**{d['num_ratings'].corr(d['avg_rating'], method='spearman'):.2f}**")
    else:
        st.info("ℹ️ Chart 4 (rating vs number of ratings) is hidden: no 'number of ratings' information.")


def top_table(d: pd.DataFrame, meta: dict) -> pd.DataFrame:
    t = pd.DataFrame({"Rank": range(1, len(d) + 1), "Movie": d["title"].values,
                      "Genre": d["genres"].str.replace("|", ", ", regex=False).values,
                      "Year": d["year"].map(lambda y: "" if pd.isna(y) else int(y)).values,
                      "Avg Rating": d["avg_rating"].round(2).values})
    if meta["has_counts"]:
        t["Ratings"] = d["num_ratings"].values
        t["Weighted Rating"] = d["weighted_rating"].round(3).values
    t["Popularity"] = d["popularity"].round(2).values
    if not meta["has_year"]:
        t = t.drop(columns="Year")
    return t


def hbar(d, x, title, color):
    d = d.copy()
    d["_label"] = d["title"].str.slice(0, 45)
    fig = px.bar(d, x=x, y="_label", orientation="h", color_discrete_sequence=[color],
                 hover_name="title", hover_data={"_label": False, "avg_rating": ":.2f"})
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(xaxis_title=x.replace("_", " ").title())
    return style_fig(fig, title, 420)


def section_top(df, meta, n, m, C):
    st.subheader("🏆 Top Movies")
    if meta["has_counts"]:
        st.info(f"**Why a weighted rating?** A movie with one 5-star vote should not beat a movie with 10,000 votes "
                f"averaging 4.6. So we use  **WR = v/(v+m) × R + m/(v+m) × C**  where **R** = movie's average rating, "
                f"**v** = its number of ratings, **m** = minimum ratings required (currently **{m:,.0f}**), "
                f"**C** = overall average rating (currently **{C:.2f}**). Movies with few ratings are pulled toward C.")
    else:
        st.warning("⚠️ No 'number of ratings' column – the weighted rating cannot be calculated, so movies are "
                   "ranked by their plain average rating (small-sample bias is possible).")
    tabs = st.tabs(["⭐ Top rated", "📝 Most rated", "🔥 Most popular"])
    with tabs[0]:
        top = df.sort_values(["weighted_rating", "num_ratings"], ascending=False).head(n)
        show(hbar(top, "weighted_rating" if meta["has_counts"] else "avg_rating",
                  f"Top {n} by {'weighted' if meta['has_counts'] else 'average'} rating", RED))
        show_df(top_table(top, meta))
    with tabs[1]:
        if meta["has_counts"]:
            top = df.sort_values("num_ratings", ascending=False).head(n)
            show(hbar(top, "num_ratings", f"Top {n} most-rated movies", GOLD))
            show_df(top_table(top, meta))
        else:
            st.info("ℹ️ Unavailable: no 'number of ratings' information.")
    with tabs[2]:
        top = df.sort_values("popularity", ascending=False).head(n)
        if df["popularity"].notna().any():
            if meta["pop_proxy"]:
                st.caption("No popularity column found – popularity = number of ratings.")
            show(hbar(top, "popularity", f"Top {n} most popular movies", "#8E7CC3"))
            show_df(top_table(top, meta))
        else:
            st.info("ℹ️ Unavailable: no popularity information.")


def section_genres(gs, meta):
    st.subheader("🎭 Genre Analysis")
    if gs.empty:
        st.warning("⚠️ No genre information in this dataset (or none in the current selection).")
        return
    pop_by = "total_ratings" if meta["has_counts"] else "movies"
    pop_g = gs.sort_values(pop_by, ascending=False).iloc[0]
    best = best_genre(gs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Most popular genre", pop_g["genre"], help=f"Based on {pop_by.replace('_', ' ')}")
    c2.metric("Highest-rated genre", best["genre"], f"{best['avg_rating']:.2f} avg")
    c3.metric("Genres found", len(gs))
    a, b = st.columns(2)
    with a:
        d = gs.sort_values("movies", ascending=False).head(30)
        fig = px.bar(d, x="movies", y="genre", orientation="h", color_discrete_sequence=[RED])
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(xaxis_title="Number of movies")
        show(style_fig(fig, "Number of movies by genre", 440))
    with b:
        size = "total_ratings" if meta["has_counts"] and gs["total_ratings"].notna().any() else "movies"
        fig = px.scatter(gs, x="movies", y="avg_rating", size=size, color="genre", size_max=45,
                         hover_data={"movies": True, "avg_rating": ":.2f"})
        fig.update_layout(xaxis_title="Number of movies", yaxis_title="Average rating")
        show(style_fig(fig, f"Genre map (bubble size = {size.replace('_', ' ')})", 440))
    t = gs.copy()
    if meta["has_counts"] and t["total_ratings"].notna().any():
        t["share"] = (t["total_ratings"] / t["total_ratings"].sum() * 100).round(1)
    table = pd.DataFrame({"Genre": t["genre"], "Movies": t["movies"], "Avg Rating": t["avg_rating"].round(3)})
    if "share" in t:
        table["Total Ratings"] = t["total_ratings"].astype("int64")
        table["Share of ratings %"] = t["share"]
    show_df(table.sort_values("Movies", ascending=False))


def section_trends(df, meta):
    st.subheader("📅 Movie Trends Over Time")
    d = df.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    yr = d.groupby("year").agg(movies=("key", "size"), avg_rating=("avg_rating", "mean"),
                               total_ratings=("num_ratings", "sum")).reset_index()
    a, b = st.columns(2)
    with a:
        fig = px.bar(yr, x="year", y="movies", color_discrete_sequence=[RED])
        fig.update_layout(xaxis_title="Release year", yaxis_title="Movies released")
        show(style_fig(fig, "Movies released per year", 360))
    with b:
        fig = px.line(yr, x="year", y="avg_rating", markers=True, hover_data={"movies": True},
                      color_discrete_sequence=[GOLD])
        fig.update_layout(xaxis_title="Release year", yaxis_title="Mean of movie average ratings")
        show(style_fig(fig, "Average rating by release year", 360))
    if meta["has_counts"] and yr["total_ratings"].gt(0).any():
        fig = px.bar(yr, x="year", y="total_ratings", color_discrete_sequence=["#8E7CC3"])
        fig.update_layout(xaxis_title="Release year", yaxis_title="Total ratings received")
        show(style_fig(fig, "Number of ratings received by movies of each release year", 340))
    st.markdown("**Most popular movie of each year**")
    best = d.loc[d.groupby("year")["popularity"].apply(lambda s: s.fillna(-1).idxmax())]
    t = pd.DataFrame({"Year": best["year"], "Movie": best["title"], "Avg Rating": best["avg_rating"].round(2),
                      "Ratings": best["num_ratings"], "Popularity": best["popularity"].round(2)})
    show_df(t.sort_values("Year", ascending=False), height=300)


def section_users(base_f, overall_avg, top_n):
    st.subheader("👥 User Rating Analysis")
    st.caption("User IDs in the data are anonymous identifiers; no personal information is shown.")
    u = base_f.groupby("user")["rating"].agg(n_ratings="size", avg_rating="mean").reset_index()
    a, b = st.columns(2)
    with a:
        fig = px.histogram(u, x="n_ratings", nbins=40, log_y=True, color_discrete_sequence=[RED])
        fig.update_layout(xaxis_title="Ratings given by a user", yaxis_title="Number of users (log)")
        show(style_fig(fig, "Number of ratings per user", 360))
    with b:
        fig = px.histogram(u, x="avg_rating", nbins=30, color_discrete_sequence=[GOLD])
        fig.add_vline(x=overall_avg, line_dash="dash", line_color="#ffffff", annotation_text="overall avg")
        fig.update_layout(xaxis_title="Average rating given by a user", yaxis_title="Number of users")
        show(style_fig(fig, "Average rating given by users", 360))
    c, d_ = st.columns(2)
    with c:
        act = u.sort_values("n_ratings", ascending=False).head(top_n).copy()
        act["label"] = "User " + act["user"].astype(str)
        fig = px.bar(act, x="n_ratings", y="label", orientation="h", color_discrete_sequence=["#4FC3F7"],
                     hover_data={"avg_rating": ":.2f", "label": False})
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(xaxis_title="Ratings given")
        show(style_fig(fig, f"Top {top_n} most active users", 400))
    with d_:
        mu, sd = u["avg_rating"].mean(), u["avg_rating"].std(ddof=0)
        hi, lo = mu + 0.5 * sd, mu - 0.5 * sd
        u["Behaviour"] = np.where(u["avg_rating"] > hi, "Generous raters",
                                  np.where(u["avg_rating"] < lo, "Critical raters", "Balanced raters"))
        fig = px.pie(u, names="Behaviour", hole=0.55, color="Behaviour",
                     color_discrete_map={"Generous raters": GOLD, "Balanced raters": "#8E7CC3", "Critical raters": RED})
        show(style_fig(fig, "Rating behaviour of users", 400))
        st.caption(f"Generous: average above {hi:.2f}; critical: below {lo:.2f} (mean ± 0.5 std of user averages).")
    s = u.sample(min(len(u), 5000), random_state=42) if len(u) > 5000 else u
    fig = px.scatter(s, x="n_ratings", y="avg_rating", log_x=True, opacity=0.6, color_discrete_sequence=[RED],
                     hover_data={"user": True, "n_ratings": True, "avg_rating": ":.2f"})
    fig.add_hline(y=overall_avg, line_dash="dash", line_color=GOLD, annotation_text="overall average")
    fig.update_layout(xaxis_title="Number of ratings (log)", yaxis_title="User's average rating")
    show(style_fig(fig, "User average rating vs overall average", 400))


def section_recommend(movies_all, df, meta, sig, filters_key):
    st.subheader("🤖 Movie Recommendation System")
    if len(df) < 2:
        st.warning("⚠️ At least two movies must match the filters to produce recommendations.")
        return
    opts_df = df.sort_values("popularity", ascending=False, na_position="last").head(20000)
    options = opts_df["title"].tolist()
    if len(df) > 20000:
        st.caption("The list shows the 20,000 most popular movies; use the sidebar search to narrow down.")
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Select (or type to search) a movie", options, index=0, key=f"rec_movie_{filters_key}")
    n = c2.radio("How many?", [5, 10], horizontal=True, key="rec_n")
    sel_key = df.loc[df["title"] == choice, "key"].iloc[0]
    table, method = recommend(movies_all, sel_key, set(df["key"]), n, sig)
    st.markdown(f"**Method:** {method}")
    if table.empty:
        st.warning("⚠️ No similar movies found inside the current filters. Try relaxing the filters.")
        return
    show_df(table)
    st.download_button("⬇️ Download recommendations (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="recommendations.csv", mime="text/csv")
    with st.expander("How does it work? (simple language)"):
        st.markdown(
            "1. Every movie is turned into a short text: its genres (and description/keywords/cast if the "
            "dataset has them).\n"
            "2. **TF-IDF** converts that text into numbers. Words that are rare and informative get a higher "
            "weight than words that appear everywhere.\n"
            "3. **Cosine similarity** measures the angle between two movies' number-vectors: 1 = identical "
            "content, 0 = nothing in common.\n"
            "4. The movies with the highest similarity to your selection are shown (ties → higher weighted rating).")


def section_bigdata(raw, meta, base, file_size, pct, top_n):
    st.subheader("📊 Big Data Insights")
    rep = meta["report"]
    mem_mb = raw.memory_usage(deep=True).sum() / 1e6
    vel = rep.get("velocity")
    st.markdown("Movie-rating data is a classic Big Data problem: platforms such as Netflix or MovieLens collect "
                "**billions of ratings** from millions of users. Below are the 5 Vs, measured on *your* dataset.")
    cards = [
        ("Volume", f"<b>{rep['raw_rows']:,}</b> rows × {rep['raw_cols']} columns<br>File ≈ {file_size/1e6:.2f} MB, "
                   f"in memory ≈ {mem_mb:.1f} MB.<br>Real platforms hold billions of rows → needs distributed storage."),
        ("Velocity", (f"Ratings span <b>{vel['start']} → {vel['end']}</b> ({vel['days']:,} days) ≈ "
                      f"<b>{vel['per_day']:,.1f}</b> ratings/day.<br>" if vel else
                      "No timestamp column detected, so the arrival rate cannot be measured.<br>")
                     + "New ratings arrive continuously → stream processing (Spark Streaming / Kafka)."),
        ("Variety", f"Numeric (ratings, years), categorical (genres), text (titles"
                    f"{', ' + ', '.join(meta['text_cols']) if meta['text_cols'] else ''}) and time data.<br>"
                    f"Genres are <b>multi-valued</b> ('Action|Comedy') and need splitting."),
        ("Veracity", f"Duplicates removed: <b>{rep['duplicates_removed']:,}</b><br>Invalid rows removed: "
                     f"<b>{rep['invalid_rows_removed']:,}</b><br>Missing cells: <b>{rep['missing_pct']:.2f}%</b><br>"
                     f"Missing genres: <b>{rep['missing_genre_rows']:,}</b> rows"),
        ("Value", "Better recommendations, fair rankings (weighted rating), genre/trend insights and user-behaviour "
                  "understanding → higher engagement and smarter content decisions."),
    ]
    cols = st.columns(5)
    for col, (name, body) in zip(cols, cards):
        col.markdown(f"<div class='vcard'><b>{name}</b><br><span style='font-size:.85rem'>{body}</span></div>",
                     unsafe_allow_html=True)

    st.markdown("#### Where Apache Spark / PySpark helps")
    show_df(pd.DataFrame({
        "Task in this project": ["Reading & cleaning ratings", "Average rating & count per movie (group-by)",
                                 "Genre statistics (explode + group-by)", "Weighted rating ranking",
                                 "Recommendations at scale"],
        "Pandas (this dashboard)": ["read_csv + drop_duplicates", "df.groupby('movie').agg(...)",
                                    "explode + groupby", "vectorised formula", "TF-IDF + cosine similarity"],
        "Spark equivalent": ["spark.read.csv + dropDuplicates", "groupBy('movie').agg(avg, count)",
                             "explode(split()) + groupBy", "withColumn(expression)", "ALS collaborative filtering (MLlib)"]}))

    st.markdown("#### 🔥 Live PySpark processing")
    has_spark, has_java = spark_status()
    if not has_spark:
        st.info("PySpark is not installed. To enable this panel run `pip install pyspark` (Java 8/11/17 required) "
                "and restart the app. No Spark results are shown until it really runs.")
        return
    if not has_java:
        st.warning("PySpark is installed but Java was not found on the PATH. Install JDK 11 or 17 and restart.")
        return
    if st.checkbox("Run PySpark analysis on the cleaned dataset (all rows, ignores sidebar filters)"):
        try:
            res = run_pyspark_analysis(base, meta["rating_level"], pct, top_n)
        except Exception as exc:   # noqa: BLE001
            st.error(f"PySpark could not run: {exc}")
            return
        c = st.columns(4)
        c[0].metric("Records read", f"{res['n_raw']:,}")
        c[1].metric("After preprocessing", f"{res['n_clean']:,}")
        c[2].metric("Movies (group-by)", f"{res['n_movies']:,}")
        c[3].metric("Spark time", f"{res['seconds']:.1f}s")
        st.caption(f"Overall average rating C = {res['C']:.3f} (mean of movie averages).")
        if "top" in res:
            st.markdown(f"**Top {top_n} movies by weighted rating (computed in Spark, m = {res['m']:.0f})**")
            show_df(res["top"].round(3))
        st.markdown("**Genre aggregation (computed in Spark)**")
        show_df(res["genre"].round(3))


def section_insights(insights):
    st.subheader("💡 Key Insights")
    st.caption("Generated automatically from the currently filtered data.")
    for text in insights:
        st.markdown(f"<div class='insight'>{text}</div>", unsafe_allow_html=True)


def summary_report(kpis, f, insights, top_df, label) -> str:
    """Plain-text/Markdown report for download."""
    plain = [re.sub(r"<[^>]+>", "", html.unescape(i)) for i in insights]
    lines = ["# Movie Rating, Recommendation & Analysis – Summary Report", f"Data source: {label}", "",
             "## Key metrics"] + [f"- {k}: {v}" for k, v in kpis] + ["", "## Active filters",
             f"- Genres: {', '.join(f['genres']) or 'all'}", f"- Rating range: {f['rating'][0]:.1f} – {f['rating'][1]:.1f}",
             f"- Year range: {f['year'] if f['year_active'] else 'all'}", f"- Minimum ratings: {f['min_ratings']}",
             f"- Title search: {f['search'] or '—'}", "", "## Key insights"] + [f"- {p}" for p in plain]
    lines += ["", "## Top movies", "```", top_df.to_string(index=False), "```"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 8. MAIN MOVIEMATE DASHBOARD
# ----------------------------------------------------------------------------
def render_movie_cards(df, nlp_feedback):
    st.success(nlp_feedback)
    if df.empty:
        st.warning("⚠️ No movies matched your preferences exactly. Try a broader search!")
        return
        
    st.markdown("### 🎬 Recommended for You")
    for idx, row in df.head(10).iterrows():
        title = row['title']
        rating = f"{row['avg_rating']:.2f}"
        year = int(row['year']) if pd.notna(row['year']) else 'N/A'
        genres = str(row['genres']).replace("|", ", ")
        cnt = f"{row['num_ratings']:,}" if pd.notna(row['num_ratings']) else 'N/A'
        ws = f"{row['weighted_rating']:.2f}" if 'weighted_rating' in row else 'N/A'
        
        score_val = row.get('recommendation_score', 0)
        why = f"Matches your preferences and has a strong rating profile."
        
        html = f"""
        <div class="movie-card">
            <div class="mc-title">{title}</div>
            <div class="mc-meta">⭐ {rating}/5 &nbsp;|&nbsp; 🎭 {genres} &nbsp;|&nbsp; 📅 {year} &nbsp;|&nbsp; 🔥 {cnt} Ratings &nbsp;|&nbsp; 📊 Score: {ws}</div>
            <div class="mc-why">💡 <b>Why recommended:</b> {why}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()
    
    # 1. READ FIXED DATASET
    try:
        raw = pd.read_csv("movies.csv")
    except Exception as e:
        st.error(f"❌ Failed to load movies.csv. Ensure it is in the same directory. {e}")
        st.stop()
    detected = dict(movie_id="movieId", title="title", genre="genres", rating="rating", user_id="userId", timestamp="timestamp")
    # For synthetic data:
    text_cols = []
    try:
        prep = prepare_data(raw, "fixed", tuple(sorted(detected.items())), tuple(text_cols))
    except Exception as e:
        st.error(f"❌ Failed to prepare data. {e}")
        st.stop()
        
    movies, base, meta = prep["movies"], prep["base"], prep["meta"]
    movies_w, m, C = add_weighted_rating(movies, meta["has_counts"], 80)
    
    # NLP ENGINE SECTION
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = ""
    if 'current_query' not in st.session_state:
        st.session_state['current_query'] = ""

    st.markdown("### 🍿 What are you in the mood for?")
    
    def set_query(q):
        st.session_state['current_query'] = q
        
    query = st.text_input("Enter natural-language preference", 
                         placeholder="Suggest me funny movies with high ratings from 2020",
                         value=st.session_state['current_query'])
                         
    col1, _, _ = st.columns([1, 2, 2])
    with col1:
        submit = st.button("🔍 Find Movies", use_container_width=True)
        
    st.markdown("**Quick Search:**")
    qc = st.columns(8)
    quick_queries = ["😂 Comedy", "🔥 Action", "❤️ Romance", "👻 Horror", "🕵️ Thriller", "🚀 Sci-Fi", "⭐ Top Rated", "🎬 Popular"]
    for i, q in enumerate(quick_queries):
        if qc[i].button(q, key=f"q_{i}", use_container_width=True):
            set_query(q[2:].strip() + " movies")
            st.rerun()
            
    if submit or st.session_state['current_query']:
        active_query = query if submit else st.session_state['current_query']
        prefs = nlp_engine.parse_query(active_query)
        feedback = nlp_engine.generate_nlp_feedback(prefs)
        
        with st.expander("🧠 **NLP Query Processing (Click to view extraction)**"):
            st.markdown(f"**User Query:** {active_query}")
            st.markdown("**Extracted Preferences:**")
            st.json(prefs)
            
        recommended_df = recommendation.get_recommendations(movies_w, prefs)
        render_movie_cards(recommended_df, feedback)
    
    st.markdown("---")
    st.header("📊 Explore Analytics & Big Data Insights")
    
    tab_names = ["⭐ Ratings", "🏆 Top Movies", "🎭 Genres", "📅 Trends", "👥 Users", "📊 Big Data"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    
    gs = genre_stats(movies_w, meta["has_counts"])
    
    with tabs["⭐ Ratings"]:
        safe("Rating analysis", section_ratings, movies_w, base, meta, gs)
    with tabs["🏆 Top Movies"]:
        safe("Top movies", section_top, movies_w, meta, 10, m, C)
    with tabs["🎭 Genres"]:
        safe("Genre analysis", section_genres, gs, meta)
    with tabs["📅 Trends"]:
        safe("Trend analysis", section_trends, movies_w, meta)
    with tabs["👥 Users"]:
        if meta["has_users"]:
            safe("User analysis", section_users, base, float(base["rating"].mean()), 10)
    with tabs["📊 Big Data"]:
        safe("Big data insights", section_bigdata, raw, meta, base, 1400000, 80, 10)
        
if __name__ == "__main__":
    main()
