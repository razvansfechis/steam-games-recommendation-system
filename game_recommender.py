import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def build_recommendation_system(df):
    # Clean the tags by removing "," between them and adding " ".
    clean_tags = df['tags'].astype(str).str.replace(",", " ", regex=False).str.strip()

    # Create a new DataFrame column called description_and_tags. It will be used for the TF-IDF algorithm
    df['description_and_tags'] = clean_tags + " " + df['cleaned_description'].astype(str)

    # Implement the TF-IDF and remove stop words like "the", "is", "and", etc.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description_and_tags'])

    # Transform the game's data (positive_ratings_percentile, avg_no_owners, median_playtime) to data between 0 and 1
    # and fill the missing data with 0
    games_data = ['positive_ratings_percentile', 'avg_no_owners', 'median_playtime']
    scaled_games_data = MinMaxScaler().fit_transform(df[games_data].fillna(0))

    # This creates a score from the scaled games data above with the dot product of the assigned weight for
    # each type of data
    df['market_likeness_games_data'] = np.dot(scaled_games_data, np.array([0.40, 0.30, 0.30]))

    return tfidf_matrix, df


def get_recommendations(selected_game, df, tfidf_matrix, top_n=5):
    # Find the index (location) of the selected game
    selected_game_index = df[df['name'].str.lower() == selected_game.lower()].index[0]

    # Computes the Cosine Similarity between the selected game's index and the rest of the games and use flatten to
    # convert the result from a 2D array to a 1D array
    tags_description_similarity = linear_kernel(tfidf_matrix[selected_game_index], tfidf_matrix).flatten()

    # Get the values for the market similarity of games
    market_likeness_similarity = df['market_likeness_games_data'].values

    # Give an importance score for each value. In here I deemed tags_description_similarity more than twice as important
    # as the market likeness similarity
    df['score'] = (tags_description_similarity * 0.70) + (market_likeness_similarity * 0.30)

    selected_games = df[df['name'].str.lower() != selected_game.lower()]

    # return the values to be used in the streamlit_application.py
    return selected_games.sort_values(by='score', ascending=False).head(top_n)[['name', 'header_image_url',
                                                                                'short_description']]