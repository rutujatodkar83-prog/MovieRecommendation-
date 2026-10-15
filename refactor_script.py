import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add new imports at the top
if "nlp_engine" not in content:
    content = content.replace("import streamlit as st", "import streamlit as st\nimport nlp_engine\nimport recommendation")

# 2. Add MovieMate CSS to CSS definition
new_css = """
.movie-card {
    background: linear-gradient(160deg, #171b26, #12151e);
    border: 1px solid #262c3d;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    transition: 0.2s;
}
.movie-card:hover {
    border-color: #E50914;
    transform: translateY(-2px);
}
.mc-title { font-size: 1.25rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
.mc-meta { font-size: 0.85rem; color: #9aa3b8; margin-bottom: 8px; }
.mc-why { font-size: 0.9rem; color: #dfe3ee; background: #0f1117; padding: 10px; border-radius: 6px; border-left: 3px solid #E50914; }
.stTextInput > div > div > input {
    font-size: 1.2rem;
    padding: 15px;
}
"""
if ".movie-card" not in content:
    content = content.replace("</style>\"\"\"", new_css + "</style>\"\"\"")

# 3. Replace render_header
header_replacement = """def render_header():
    st.markdown(
        "<div class='hero'><h1>🎬 MovieMate</h1>"
        "<div class='sub'>Natural Language Movie Recommendation & Big Data Analytics</div>"
        "<p>Discover movies using natural-language preferences and explore insights from large-scale movie rating data.</p></div>", unsafe_allow_html=True)
"""
content = re.sub(r'def render_header\(\):.*?(?=def |\n#)', header_replacement, content, flags=re.DOTALL)

# 4. Replace MAIN entirely
new_main = """
# ----------------------------------------------------------------------------
# 8. MAIN MOVIEMATE DASHBOARD
# ----------------------------------------------------------------------------
def render_movie_cards(df, nlp_feedback):
    st.success(nlp_feedback)
    if df is empty:
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
        
        html = f\"\"\"
        <div class="movie-card">
            <div class="mc-title">{title}</div>
            <div class="mc-meta">⭐ {rating}/5 &nbsp;|&nbsp; 🎭 {genres} &nbsp;|&nbsp; 📅 {year} &nbsp;|&nbsp; 🔥 {cnt} Ratings &nbsp;|&nbsp; 📊 Score: {ws}</div>
            <div class="mc-why">💡 <b>Why recommended:</b> {why}</div>
        </div>
        \"\"\"
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
        
    detected = dict(movie_id="movieId", title="title", genre="genres", rating="rating", num_ratings="num_ratings", user_id="userId", timestamp="timestamp")
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
"""

# Replace the old main function (from line "def main():" to end)
content = re.sub(r'def main\(\):.*', new_main, content, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("app.py refactored successfully.")
