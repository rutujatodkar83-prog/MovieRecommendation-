import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

new_main = """def main():
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
    gs = genre_stats(movies_w, meta["has_counts"])
    
    # 2. TOP LEVEL NAVIGATION
    tab_home, tab_analytics, tab_nlp, tab_bigdata, tab_about = st.tabs([
        "🏠 Home", "📊 Analytics", "🧠 NLP / How It Works", "📈 Big Data", "ℹ️ About"
    ])
    
    # ------------------ HOME TAB ------------------
    with tab_home:
        if 'last_query' not in st.session_state:
            st.session_state['last_query'] = ""
        if 'current_query' not in st.session_state:
            st.session_state['current_query'] = ""

        st.markdown("### 🍿 What are you in the mood for?")
        
        def set_query(q):
            st.session_state['current_query'] = q
            
        query = st.text_input("Enter natural-language preference", 
                             placeholder="Suggest me funny movies with high ratings from 2020",
                             value=st.session_state['current_query'],
                             label_visibility="collapsed")
        
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
            
            recommended_df = recommendation.get_recommendations(movies_w, prefs)
            render_movie_cards(recommended_df, feedback)

    # ------------------ ANALYTICS TAB ------------------
    with tab_analytics:
        st.markdown("## 📊 Movie Analytics")
        st.markdown("#### Explore insights from the movie rating dataset.")
        
        # KPI Cards specific requested elements
        total_ratings = raw["rating"].count() if "rating" in raw else 0
        total_movies = len(movies_w)
        total_users = raw["userId"].nunique() if "userId" in raw else 0
        avg_rating = movies_w["avg_rating"].mean()
        total_genres = len(gs)
        pop_movie = movies_w.sort_values("num_ratings", ascending=False).iloc[0]["title"] if not movies_w.empty else "N/A"
        
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("🎬 Total Movies", f"{total_movies:,}")
        c2.metric("👥 Total Users", f"{total_users:,}")
        c3.metric("⭐ Average Rating", f"{avg_rating:.2f}")
        c4.metric("📝 Total Ratings", f"{total_ratings:,}")
        c5.metric("🎭 Total Genres", f"{total_genres:,}")
        c6.metric("🔥 Most Rated Movie", pop_movie)
        
        st.markdown("---")
        
        atabs = st.tabs(["⭐ Ratings", "🏆 Top Movies", "🎭 Genres", "📅 Trends", "👥 Users"])
        with atabs[0]: safe("Rating analysis", section_ratings, movies_w, base, meta, gs)
        with atabs[1]: safe("Top movies", section_top, movies_w, meta, 10, m, C)
        with atabs[2]: safe("Genre analysis", section_genres, gs, meta)
        with atabs[3]: safe("Trend analysis", section_trends, movies_w, meta)
        with atabs[4]: 
            if meta["has_users"]:
                safe("User analysis", section_users, base, float(base["rating"].mean()), 10)
    
    # ------------------ NLP TAB ------------------
    with tab_nlp:
        st.markdown(\"\"\"
## 🧠 NLP / How It Works
Understanding how MovieMate processes your natural-language queries.

### 1. User Query
> *"Suggest me funny movies with high ratings from 2020."*

### 2. Preference Extraction (NLP Processing)
Through heuristics, Regex and Keyword Mapping, we extract:
- **Genre:** Comedy
- **Rating:** High
- **Year:** 2020

### 3. Execution Pipeline
`Input Text` ➔ `Preference Extraction` ➔ `Dataset Filtering` ➔ `Movie Ranking` ➔ `Recommendations`

---
### 🔍 Query Examples & Fallbacks
- **"Suggest me funny movies."** ➔ Maps to `Comedy`.
- **"Give me highly rated action movies."** ➔ Filters for `Action` + `High Rating`.
- **"Suggest movies from the 2010s."** ➔ Filters constraint `Year 2010–2019`.
- **"Suggest movies like Hidden Kingdom."** ➔ Uses `TF-IDF / Cosine Similarity` against movie titles & genres.
- **"Give me movies starring Tom Cruise."** ➔ The backend detects an 'Actor' intent, and gracefully explains that actor information is unvailable in the current dataset.
        \"\"\")

    # ------------------ BIG DATA TAB ------------------
    with tab_bigdata:
        st.markdown(\"\"\"
## 📈 Big Data Processing Pipeline
This project applies Big Data analytics methodologies to process movie ratings efficiently.

**Pipeline flow:**
`CSV Dataset` ➔ `Data Ingestion` ➔ `Data Cleaning` ➔ `Transformation` ➔ `Distributed/Batch Processing` ➔ `Aggregation` ➔ `Recommendation`

---
### 🔎 The 5Vs of Big Data
* **Volume**: Thousands of records forming large-scale relational matrices.
* **Variety**: User IDs, movie IDs, genres, float ratings and timestamps.
* **Velocity**: Movie ratings can be continuously generated. *(Note: This project currently performs batch processing on a movie-rating dataset. Real-time streaming can be implemented as future scope).*
* **Veracity**: Raw ratings contain missing titles, duplicate actions or inconsistent distributions which are handled in the Cleaning phase.
* **Value**: Discovering patterns and relationships to provide value through Movie Recommendations.

---
### 🛠️ Big Data Operations
* **Data Ingestion**: Loading large CSV files safely chunk-by-chunk.
* **Data Cleaning / Filtering**: Removing sparse genres & irrelevant entries.
* **GroupBy & Aggregation**: Calculating `mean` ratings and `count` distribution per movie.
* **Transformation**: Feature engineering the Bayesian Weighted Rating formula.
        \"\"\")
        
        st.markdown("---")
        safe("Big data insights", section_bigdata, raw, meta, base, 1400000, 80, 10)

    # ------------------ ABOUT TAB ------------------
    with tab_about:
        st.markdown(\"\"\"
## ℹ️ About
**MovieMate — Natural Language Movie Recommendation & Big Data Analytics System**

### Objective
This project integrates Big Data Analytics, Natural Language Processing, and a Content-Based Recommendation System through Interactive Data Visualizations.

### Technology Stack
* **Python**: Core algorithmic logic.
* **Streamlit**: Web rendering engine.
* **Pandas / NumPy**: In-memory data manipulation and grouping.
* **Plotly**: Interactive visualization and charting.
* **Scikit-learn**: TF-IDF Vectorization and Cosine Similarity.
* **PySpark** (Optional): Scale-out cluster analytics for massive multi-node datasets.

### Dataset Features (`movies.csv`)
- `userId`: Identifier for users.
- `movieId`: Integer movie index.
- `title`: Extracted string.
- `genres`: Categorical features (pipe separated).
- `rating`: Float format target metric.
- `timestamp`: Unix time.

### Dataset Limitations
* No Actor or Director information available.
* No exact revenue/popularity metric (Popularity is derived directly from the **Number of ratings per movie**).
* No descriptive Plot/Keyword information (Recommendations are heavily weighted on Genre TF-IDF + User Preferences).
        \"\"\")

if __name__ == "__main__":
    main()
"""

# Replace main function
content = re.sub(r'def main\(\):.*', new_main, content, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("app.py refactored successfully.")
