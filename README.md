# 🎬 Movie Rating, Recommendation & Analysis Dashboard
**Big Data Analytics – ISE Project**

## 1. Objective
Analyse a movie dataset with Big Data concepts and present insights about ratings, popularity, genres,
rating distribution, top movies, user rating patterns, trends and movie recommendations in one
interactive dashboard.

## 2. Features
- Upload any movie CSV (MovieLens-style or similar). Columns are **auto-detected**, never assumed;
  a *Column mapping* box lets you correct a wrong guess.
- Optional second upload for MovieLens split files (`movies.csv` + `ratings.csv`, joined on movie ID).
- Cleaning: duplicate rows, repeated (user, movie) ratings, missing values, invalid ratings.
- KPI cards, sidebar filters (genre, rating, year, minimum ratings, title search, Top-N) + **Reset Filters**.
- Rating analysis (4 charts), Top Movies with **weighted rating**, Genre analysis, Trends over time.
- Content-based **recommendation system** (TF-IDF + cosine similarity) with a simple fallback.
- User rating analysis, **Big Data Insights** (5 Vs measured on your data), optional **live PySpark** panel.
- Auto-generated **Key Insights** (computed from the data, never hard-coded).
- CSV / Markdown exports. Plotly charts: hover, zoom, legend toggle, PNG download (camera icon).
- Sections whose columns are missing are hidden/disabled with a clear warning – the app does not crash.

## 3. Technologies
Python · Streamlit · Pandas / NumPy · Plotly · scikit-learn · (optional) PySpark

## 4. Dataset description
Minimum: a **rating** column plus a **movie title or movie ID** column. Everything else is optional:

| Role | Example column names recognised | Enables |
|---|---|---|
| Movie ID | movieId, item_id, id | merging files, unique movies |
| Title | title, movie_title, original_title | labels, search |
| Genre | genres, genre (`A\|B`, `A, B`, or JSON style) | genre analysis |
| Rating | rating, vote_average, score | everything |
| Number of ratings | vote_count, num_ratings | weighted rating, most-rated |
| User ID | userId, user_id | user analysis |
| Year | year, release_date (or `Title (1995)`) | trends, year filter |
| Popularity | popularity | popularity ranking |
| Overview/keywords/cast/director | any column containing these words | richer recommendations |

Two layouts are handled automatically:
1. **Rating-level** – one row per rating (MovieLens `ratings.csv`, or ratings joined with movies). The app
   aggregates it to one row per movie (average rating, number of ratings).
2. **Movie-level** – one row per movie with an average rating (e.g. TMDB `vote_average`, `vote_count`).

The bundled `movies.csv` is **synthetic demo data** created by `generate_sample_data.py` so the project runs
immediately. Use real data (e.g. MovieLens from https://grouplens.org/datasets/movielens/) for your report.

## 5. Big Data concepts used
- **Volume / Velocity / Variety / Veracity / Value** – measured on the uploaded dataset in the *Big Data* tab.
- **Group-by aggregation**, **explode** of multi-valued genres, **data cleaning** – the same operations are
  re-run in **PySpark** (read → clean → group-by movie → weighted rating → group-by genre) and the results
  match the Pandas results. Only real Spark output is displayed.

## 6. Recommendation methodology
1. Each movie becomes a text "document": genre tokens (+ overview / keywords / cast if present).
2. **TF-IDF** turns the document into a numeric vector.
3. **Cosine similarity** compares the selected movie with all others; the top 5/10 are shown
   (ties are broken by weighted rating).
4. Fallback when no genre/text exists: movies with the closest average rating and popularity.

Weighted rating: `WR = v/(v+m)·R + m/(v+m)·C` (R = movie average, v = its ratings, m = minimum ratings
(percentile of counts, default 80th), C = overall average).

## 7. Installation
```bash
cd movie_dashboard
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install -r requirements.txt
```
Optional – live PySpark panel (needs Java 11/17/21 installed):
```bash
pip install pyspark
```

## 8. Run
```bash
streamlit run app.py
```
Open http://localhost:8501. To regenerate the demo data: `python generate_sample_data.py`.

## 9. Expected output
A dark, cinematic dashboard: hero header → six KPI cards → tabs
(Data · Ratings · Top Movies · Genres · Trends · Recommendations · Users · Big Data · Insights).
Sidebar: upload, column mapping, filters, reset button, exports.

## 10. Project structure
```
movie_dashboard/
├── app.py                    # whole dashboard (functions grouped in numbered sections)
├── movies.csv                # synthetic demo data
├── generate_sample_data.py   # creates movies.csv
├── requirements.txt
├── README.md
└── .streamlit/config.toml    # dark theme
```

## 11. Future scope
Collaborative filtering (ALS in Spark MLlib), hybrid recommender, sentiment analysis of reviews,
real-time ratings with Kafka + Spark Streaming, user login with personalised recommendations,
deployment on Streamlit Cloud / Docker with a Spark cluster.
