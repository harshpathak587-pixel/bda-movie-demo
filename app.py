import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load():
    user_ids = np.load("demo_data/user_ids.npy")
    user_vecs = np.load("demo_data/user_vecs.npy")
    item_vecs = np.load("demo_data/item_vecs.npy")
    pop = pd.read_csv("demo_data/popular_movies.csv")
    allm = pd.read_csv("demo_data/all_movies.csv")
    rated = pd.read_csv("demo_data/rated.csv")
    return user_ids, user_vecs, item_vecs, pop, allm, rated

user_ids, user_vecs, item_vecs, pop, allm, rated = load()

st.title("🎬 Movie Recommendation System")
st.caption("PySpark ALS model trained on 25 million MovieLens ratings (test RMSE 0.814)")

genres = sorted({g for s in pop["genres"] for g in s.split("|") if g != "(no genres listed)"})

st.sidebar.header("Settings")
uid = st.sidebar.number_input("User ID (1 to 1000)", min_value=1, max_value=1000, value=1, step=1)
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
min_ratings = st.sidebar.slider("Minimum ratings a movie must have", 500, 2000, 500, step=100)
genre = st.sidebar.selectbox("Genre filter", ["All"] + genres)

idx = np.where(user_ids == uid)[0][0]
user_rated = rated[rated["userId"] == uid]

c1, c2, c3 = st.columns(3)
c1.metric("User", int(uid))
c2.metric("Movies rated", len(user_rated))
c3.metric("Average rating given", round(user_rated["rating"].mean(), 2))

# predicted rating = user vector . movie vector (same as the Spark model)
df = pop.copy()
df["predicted_rating"] = np.clip(item_vecs @ user_vecs[idx], 0, 5).round(2)
df = df[~df["movieId"].isin(set(user_rated["movieId"]))]
df = df[df["num_ratings"] >= min_ratings]
if genre != "All":
    df = df[df["genres"].str.contains(genre, regex=False)]
top = df.sort_values("predicted_rating", ascending=False).head(top_n)

left, right = st.columns(2)

with left:
    st.subheader("Movies this user rated highest")
    liked = (user_rated.sort_values("rating", ascending=False).head(10)
             .merge(allm, on="movieId")[["title", "genres", "rating"]])
    st.dataframe(liked, hide_index=True, use_container_width=True)

with right:
    st.subheader("Recommended for this user")
    st.dataframe(top[["title", "predicted_rating", "num_ratings"]],
                 hide_index=True, use_container_width=True)

st.subheader("Predicted ratings")
st.bar_chart(top.set_index("title")["predicted_rating"])
