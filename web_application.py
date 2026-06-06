import streamlit as st
import pandas as pd
import base64
import game_recommender


st.set_page_config(page_title="Steam Recommender", layout="wide")
st.title("Steam Games Recommender System")


# Background (unchanged but shorter)
with open("background.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_data():
    df = pd.read_csv("data/13_steam_games_final.csv")
    return game_recommender.build_recommendation_system(df)


tfidf_matrix, df = load_data()

game = st.selectbox("Choose a game:", df["name"].unique())

if st.button("Recommend"):
    recs = game_recommender.get_recommendations(game, df, tfidf_matrix)

    st.subheader("You might also like:")

    for i, row in enumerate(recs.itertuples(), 1):
        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(row.header_image_url, width="stretch")

            with col2:
                st.markdown(f"### {i}. {row.name}")
                st.write(row.translated_short_description or "No description available")