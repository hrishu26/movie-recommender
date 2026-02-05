import streamlit as st
import pickle
import pandas as pd

# Load data FIRST (so movies/similarity exist before recommend is called)
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie, top_n=5):
    # safer lookup
    idx_list = movies.index[movies['title'] == movie].tolist()
    if not idx_list:
        return []
    movie_index = idx_list[0]

    distances = similarity[movie_index]

    # sort by similarity score
    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movie = []
    # skip first item (itself) and take top_n
    for i in movies_list[1:top_n+1]:
        movie_id=i[0]
        recommended_movie.append(movies.iloc[i[0]].title)

    return recommended_movie


st.title('Movie Recommendation System')

selected_movie_name = st.selectbox(
    'How would you like to recommend?',
    movies['title'].values
)

if st.button('Recommend'):
    recs = recommend(selected_movie_name)
    if not recs:
        st.warning("Movie not found / recommendations unavailable.")
    else:
        for m in recs:
            st.write(m)
