"""
generate_sample_data.py
-----------------------
Creates `movies.csv`: a SYNTHETIC, MovieLens-style demo dataset so the dashboard
works out-of-the-box. It is NOT real data. For real analysis, upload the
MovieLens `movies.csv` + `ratings.csv` (or any similar CSV) in the dashboard.

Run:  python generate_sample_data.py
Columns produced: userId, movieId, title, genres, rating, timestamp
(one row = one rating given by one user to one movie).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_MOVIES, N_USERS, N_RATINGS = 400, 1200, 40000

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]
# Small genre "quality" offsets so that genres differ a little in average rating
GENRE_EFFECT = {"Documentary": 0.35, "Drama": 0.25, "Crime": 0.15, "Animation": 0.15,
                "Sci-Fi": 0.05, "Adventure": 0.05, "Fantasy": 0.0, "Thriller": 0.0,
                "Romance": -0.05, "Comedy": -0.1, "Action": -0.1, "Horror": -0.3}

ADJ = ["Silent", "Last", "Hidden", "Crimson", "Broken", "Golden", "Midnight", "Lost",
       "Electric", "Fading", "Distant", "Burning", "Hollow", "Endless", "Secret", "Wild"]
NOUN = ["Horizon", "Empire", "Garden", "Signal", "Harbor", "Echo", "Voyage", "Letter",
        "Frontier", "Mirror", "Kingdom", "Orbit", "Summer", "Shadow", "Machine", "River"]

# ---- movies -----------------------------------------------------------------
titles = set()
rows = []
while len(rows) < N_MOVIES:
    year = int(rng.integers(1970, 2024))
    name = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
    if rng.random() < 0.4:
        name = f"The {name}"
    title = f"{name} ({year})"
    if title in titles:
        continue
    titles.add(title)
    g = list(rng.choice(GENRES, size=int(rng.integers(1, 4)), replace=False))
    quality = rng.normal(3.35, 0.45) + np.mean([GENRE_EFFECT[x] for x in g])
    rows.append((len(rows) + 1, title, "|".join(g), quality, year))
movies = pd.DataFrame(rows, columns=["movieId", "title", "genres", "quality", "year"])

# Popularity is heavily skewed (a few blockbusters, many niche films)
pop = rng.lognormal(mean=0.0, sigma=1.3, size=N_MOVIES)
pop = pop / pop.sum()

# ---- users & ratings ----------------------------------------------------------
user_bias = rng.normal(0, 0.35, N_USERS)
user_activity = rng.lognormal(0, 1.0, N_USERS)
user_activity = user_activity / user_activity.sum()

movie_idx = rng.choice(N_MOVIES, size=N_RATINGS, p=pop)
user_idx = rng.choice(N_USERS, size=N_RATINGS, p=user_activity)
raw = movies["quality"].values[movie_idx] + user_bias[user_idx] + rng.normal(0, 0.7, N_RATINGS)
rating = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)

ts = rng.integers(1_262_304_000, 1_700_000_000, N_RATINGS)  # 2010 .. 2023 (unix seconds)

df = pd.DataFrame({
    "userId": user_idx + 1,
    "movieId": movies["movieId"].values[movie_idx],
    "title": movies["title"].values[movie_idx],
    "genres": movies["genres"].values[movie_idx],
    "rating": rating,
    "timestamp": ts,
}).drop_duplicates(subset=["userId", "movieId"])

df.to_csv("movies.csv", index=False)
print(f"Saved movies.csv with {len(df):,} ratings, "
      f"{df.movieId.nunique()} movies and {df.userId.nunique()} users.")
