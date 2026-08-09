import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Movie Recommendation System", page_icon="🎬")

st.title("AI Movie Recommendation System")
st.write("Select a movie you enjoy, and the system will recommend movies with similar genres.")

@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("movies.csv")
    except FileNotFoundError:
        st.error(
            "movies.csv was not found. Download the MovieLens 'ml-latest-small' "
            "dataset and place movies.csv in the same folder as this app.py file."
        )
        st.stop()

    required = {"title", "genres"}
    if not required.issubset(movies.columns):
        st.error("movies.csv must contain 'title' and 'genres' columns.")
        st.stop()

    movies["genres"] = movies["genres"].fillna("")
    return movies

movies = load_data()

@st.cache_resource
def build_model(genres):
    # MovieLens separates genres with the | character.
    vectorizer = CountVectorizer(
        tokenizer=lambda text: text.split("|"),
        token_pattern=None,
        lowercase=False
    )
    genre_matrix = vectorizer.fit_transform(genres)
    similarity_matrix = cosine_similarity(genre_matrix)
    return similarity_matrix

similarity = build_model(movies["genres"])

def recommend_movies(title, number=5):
    matches = movies.index[movies["title"] == title].tolist()

    if not matches:
        return []

    selected_index = matches[0]
    scores = list(enumerate(similarity[selected_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for movie_index, score in scores:
        if movie_index == selected_index:
            continue

        recommendations.append(
            {
                "title": movies.iloc[movie_index]["title"],
                "genres": movies.iloc[movie_index]["genres"],
                "similarity": score
            }
        )

        if len(recommendations) == number:
            break

    return recommendations

selected_movie = st.selectbox(
    "Choose a movie:",
    movies["title"].sort_values().tolist()
)

number_of_recommendations = st.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("Get Recommendations", type="primary"):
    results = recommend_movies(selected_movie, number_of_recommendations)

    st.subheader("Recommended Movies")

    if not results:
        st.warning("No recommendations were found.")
    else:
        for i, result in enumerate(results, start=1):
            st.write(f"**{i}. {result['title']}**")
            st.caption(
                f"Genres: {result['genres']} | "
                f"Similarity score: {result['similarity']:.2f}"
            )

st.divider()
st.caption(
    "AI/HCI academic project using content-based filtering and cosine similarity."
)
