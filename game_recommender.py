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

    scaled = MinMaxScaler().fit_transform(
        df[["positive_ratings_percentage", "avg_no_owners", "median_playtime"]].fillna(0)
    )

    df["market_score"] = scaled @ np.array([0.2, 0.7, 0.1]) # multiply the scaled market values by the weights

    return tfidf_matrix, df


def get_recommendations(game_name, df, tfidf_matrix, top_n=5):
    game_index = df.index[df["name"].str.lower() == game_name.lower()][0]

    similarity = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).ravel() # gets the similarity and transforms
                                                                        # the array from 2D to 1D using ravel method

    score = 0.3 * similarity + 0.7 * df["market_score"].values

    df["score"] = score

    return (
        df[df["name"].str.lower() != game_name.lower()]
        .nlargest(top_n, "score")[["name", "header_image_url", "translated_short_description"]]
    )