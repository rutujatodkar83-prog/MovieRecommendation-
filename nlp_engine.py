import re

# Dataset supported genres
SUPPORTED_GENRES = {
    "Action", "Adventure", "Animation", "Comedy", "Crime", 
    "Documentary", "Drama", "Fantasy", "Horror", "Romance", 
    "Sci-Fi", "Thriller", "Mystery"
}

# Mapping common words to genres (lowercased)
GENRE_MAPPING = {
    "funny": "Comedy",
    "comedy": "Comedy",
    "humorous": "Comedy",
    "scary": "Horror",
    "horror": "Horror",
    "romantic": "Romance",
    "romance": "Romance",
    "love": "Romance",
    "action": "Action",
    "fighting": "Action",
    "thriller": "Thriller",
    "suspense": "Thriller",
    "animated": "Animation",
    "animation": "Animation",
    "cartoon": "Animation",
    "science fiction": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "crime": "Crime",
    "drama": "Drama",
    "adventure": "Adventure",
    "fantasy": "Fantasy",
    "mystery": "Mystery"
}

def parse_query(query: str):
    """
    Parses a natural language query and extracts:
    - genres: list of mapped genres
    - year: int or None
    - decade: string (like '2010s') or None
    - rating_pref: 'high' or 'specific'
    - min_rating: float or None
    - similar_to: string or None
    - popularity_pref: boolean
    - limit: int
    - unsupported: list of unsupported concepts detected (actors/time travel)
    """
    query_lower = query.lower().strip()
    
    preferences = {
        "genres": set(),
        "year": None,
        "decade": None,
        "rating_pref": None,
        "min_rating": None,
        "similar_to": None,
        "popularity_pref": False,
        "limit": 10,
        "unsupported": []
    }
    
    if not query_lower:
        return preferences

    # 1. Similarity
    sim_match = re.search(r'(?:similar to|movies like|movie like)\s+([a-z0-9 ]+)', query_lower)
    if sim_match:
        # Extract title, removing common stopwords that might follow
        parts = sim_match.group(1).split()
        title_parts = []
        for p in parts:
            if p in ["from", "with", "that", "in"]:
                break
            title_parts.append(p)
        preferences["similar_to"] = " ".join(title_parts).strip()

    # 2. Year Detection
    # Decade (e.g. 2010s, 90s)
    decade_match = re.search(r'\b(19|20)?(\d)0s\b', query_lower)
    if decade_match:
        prefix = decade_match.group(1) or ("19" if int(decade_match.group(2)) >= 5 else "20")
        preferences["decade"] = f"{prefix}{decade_match.group(2)}0s"
    else:
        # Exact Year (e.g. 2020)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query_lower)
        if year_match:
            preferences["year"] = int(year_match.group(1))
    
    if "recent" in query_lower:
        preferences["year"] = "recent"  # Special tag
    elif "old" in query_lower or "classic" in query_lower:
        preferences["year"] = "old"

    # 3. Rating Detection
    if any(w in query_lower for w in ["highly rated", "high rating", "high ratings", "good", "best", "top"]):
        preferences["rating_pref"] = "high"
        
    num_rating = re.search(r'rating above (\d(?:\.\d)?)', query_lower)
    if num_rating:
        preferences["min_rating"] = float(num_rating.group(1))

    # 4. Genre Detection
    for word, genre in GENRE_MAPPING.items():
        if re.search(r'\b' + re.escape(word) + r'\b', query_lower):
            preferences["genres"].add(genre)
            
    # 5. Quantity Detection
    qty_match = re.search(r'\b(\d+)\s+movies?\b', query_lower)
    if qty_match:
        limit = int(qty_match.group(1))
        if limit > 0 and limit <= 100:
            preferences["limit"] = limit
            
    # 6. Popularity Detection
    if any(w in query_lower for w in ["popular", "trending"]):
        preferences["popularity_pref"] = True
        
    # 7. Unsupported Concepts detection
    if "starring" in query_lower or "actor" in query_lower or "director" in query_lower or "with tom" in query_lower or "with leonardo" in query_lower:
        preferences["unsupported"].append("actor_director")
        
    if "time travel" in query_lower:
        preferences["unsupported"].append("time_travel")
        if not preferences["genres"]:
            # fallback to scifi if they asked for something like time travel
            preferences["genres"].add("Sci-Fi")

    preferences["genres"] = list(preferences["genres"])
    return preferences

def generate_nlp_feedback(prefs: dict):
    """Generate a conversational response based on extracted elements."""
    if "actor_director" in prefs.get("unsupported", []):
        return ("⚠️ Actor/Director information is not available in the current dataset. "
                "I can recommend movies based on genre, rating, year, and popularity instead! Here are some top picks.")
        
    if "time_travel" in prefs.get("unsupported", []):
        return ("I couldn't find enough information about 'time travel' in the current dataset. "
                "Here are some highly rated Sci-Fi movies from the dataset instead! 🚀")

    if prefs.get("similar_to"):
        return f"Sure! Searching for movies similar to '{prefs['similar_to'].title()}' and ranking them for you. 🍿"
        
    parts = []
    if prefs.get("rating_pref") == "high":
        parts.append("highly rated")
    elif prefs.get("min_rating"):
        parts.append(f"rated over {prefs['min_rating']}")
    
    if prefs.get("popularity_pref"):
        parts.append("popular")
        
    if prefs.get("genres"):
        parts.append("/".join(prefs["genres"]))
    
    parts.append("movies")
    
    if prefs.get("decade"):
        parts.append(f"from the {prefs['decade']}")
    elif isinstance(prefs.get("year"), int):
        parts.append(f"from {prefs['year']}")
    elif prefs.get("year") == "recent":
        parts.append("from recent years")
    elif prefs.get("year") == "old":
        parts.append("(classics/older films)")
        
    desc = " ".join(parts).title()
    if desc.strip() == "Movies":
        return "Sure! Here are some popular and highly rated movies from the dataset you might enjoy. 🍿"
    return f"Sure! Here are some {desc} you might enjoy. 🍿"
