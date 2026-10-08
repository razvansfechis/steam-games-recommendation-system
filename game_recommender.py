import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def build_recommendation_system(df):
    df = df.copy()

    df["text"] = (
            df["tags"].astype(str).str.replace(",", " ", regex=False)
            + " "
            + df["cleaned_description"].astype(str)
    )

    tfidf_matrix = TfidfVectorizer(stop_words="english").fit_transform(df["text"])

    market = df[["positive_ratings_percentage", "avg_no_owners", "median_playtime"]].fillna(0)

    # Scale owners and playtime using log so giant games with a lot of users and huge median playtime don't dominate
    market["avg_no_owners"] = np.log1p(market["avg_no_owners"])
    market["median_playtime"] = np.log1p(market["median_playtime"])

    scaled = MinMaxScaler().fit_transform(market)

    df["market_score"] = scaled @ np.array([0.2, 0.7, 0.1])  # multiply the scaled market values by the weights

    return tfidf_matrix, df


def get_recommendations(game_name, df, tfidf_matrix, top_n=5):
    game_index = df.index[df["name"].str.lower() == game_name.lower()][0]

    similarity = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).ravel()

    score = pd.Series(0.75 * similarity + 0.25 * df["market_score"].values, index=df.index)

    candidates = df["name"].str.lower() != game_name.lower()  # exclude the selected game

    top_indexes = score[candidates].nlargest(top_n).index

    return df.loc[top_indexes, ["name", "header_image_url", "translated_short_description"]]