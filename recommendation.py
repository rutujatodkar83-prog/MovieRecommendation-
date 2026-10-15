import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_scores(movies_w: pd.DataFrame, similar_to_title: str):
    """
    Given a movie title, finds it and returns a Series of similarity scores (0 to 1) 
    for all movies based on TF-IDF of genres.
    """
    movies_w = movies_w.reset_index(drop=True)
    
    # Try to find exactly or partially
    match = movies_w[movies_w['title'].str.contains(similar_to_title, case=False, na=False, regex=False)]
    if match.empty:
        return pd.Series(0.0, index=movies_w.index)
        
    pos = match.index[0]
    
    # Text docs are just genres space separated
    docs = movies_w["genres"].str.replace("|", " ", regex=False)
    
    try:
        vec = TfidfVectorizer(max_features=5000)
        matrix = vec.fit_transform(docs)
        sim = cosine_similarity(matrix[pos], matrix).ravel()
        # Set self similarity to 0 so we don't recommend the same movie
        sim[pos] = 0.0
        return pd.Series(sim, index=movies_w.index)
    except Exception:
        return pd.Series(0.0, index=movies_w.index)


def get_recommendations(movies_w: pd.DataFrame, prefs: dict):
    """
    Ranks the dataset based on extracted preferences.
    """
    df = movies_w.copy()
    
    # 1. Base filtering (Hard filters)
    if prefs.get('genres'):
        mask = pd.Series(False, index=df.index)
        for g in prefs['genres']:
            mask |= df['genres'].str.contains(g, case=False, na=False, regex=False)
        df = df[mask.fillna(False)]
        
    if prefs.get('year') == 'old':
        df = df[df['year'] < 2000]
    elif prefs.get('year') == 'recent':
        df = df[df['year'] > 2018]
    elif isinstance(prefs.get('year'), int):
        df = df[df['year'] == prefs['year']]
    elif prefs.get('decade'):
        start_year = int(prefs['decade'][:4])
        df = df[(df['year'] >= start_year) & (df['year'] < start_year + 10)]
        
    if df.empty:
        return df
        
    if prefs.get('min_rating'):
        df = df[df['avg_rating'] >= prefs['min_rating']]
        
    if df.empty:
        return df

    # 2. Re-indexing
    df = df.reset_index(drop=True)
    
    # 3. Dynamic Ranking
    # Weights
    w_genre = 0.4 if prefs.get('genres') else 0.0
    w_sim = 0.1 if prefs.get('similar_to') else 0.0
    w_rating = 0.3
    w_pop = 0.2 
    
    if prefs.get('rating_pref') == 'high':
        w_rating += 0.2
        w_pop -= 0.1
        w_genre = max(0, w_genre - 0.1)
        
    if prefs.get('popularity_pref'):
        w_pop += 0.2
        w_rating -= 0.1
        w_genre = max(0, w_genre - 0.1)
    
    # Normalize Weights
    total_w = w_genre + w_sim + w_rating + w_pop
    if total_w > 0:
        w_genre /= total_w
        w_sim /= total_w
        w_rating /= total_w
        w_pop /= total_w
        
    # Scores
    min_rating = df['weighted_rating'].min()
    max_rating = df['weighted_rating'].max()
    sr = (df['weighted_rating'] - min_rating) / (max_rating - min_rating) if max_rating > min_rating else 0.0

    min_pop = df['num_ratings'].min()
    max_pop = df['num_ratings'].max()
    sp = (df['num_ratings'] - min_pop) / (max_pop - min_pop) if max_pop > min_pop else 0.0
    
    ss = calculate_similarity_scores(df, prefs['similar_to']) if w_sim > 0.0 else 0.0
    
    # Genre match score (count overlapping genres)
    sg = 0.0
    if w_genre > 0 and prefs['genres']:
        def genre_score(g_str):
            gs = g_str.split('|')
            inter = set(gs).intersection(set(prefs['genres']))
            return len(inter) / len(prefs['genres'])
        sg = df['genres'].apply(genre_score)

    final_score = (w_rating * sr) + (w_pop * sp) + (w_genre * sg) + (w_sim * ss)
    df['recommendation_score'] = final_score
    
    return df.sort_values(by='recommendation_score', ascending=False)
