import pandas as pd
import numpy as np
import game_recommender

# Load data
df_games = pd.read_csv("../data/13_steam_games_final.csv")
df_likes = pd.read_csv("../data/data-for-testing/5-users-recommendations-final.csv")

df_games['appid'] = df_games['appid'].astype(int)
df_likes['appid'] = df_likes['appid'].astype(int)

# Build recommender
matrix, df_games = game_recommender.build_recommendation_system(df_games)

k_values = [5, 50, 500, 5000]
hits = {k: 0 for k in k_values}


# Keep only valid interactions
df_likes = df_likes[df_likes['appid'].isin(df_games['appid'])]

# Select active users (20+ interactions)
user_counts = df_likes['userid'].value_counts() # Get for each user their n.o. interactions
valid_users = user_counts[user_counts >= 20].index.to_numpy() # Filter the users and keep their ID in a numpy list

# Shuffle the selected users using a seed for replicability
np.random.seed(99)
valid_users = np.random.permutation(valid_users)[:100]

print(f"Testing {len(valid_users)} users...")

# Testing loop
for user_id in valid_users:
    try:
        user_games = df_likes[df_likes['userid'] == user_id]['appid'].tolist()

        input_game = df_games.loc[
            df_games['appid'] == user_games[0], 'name'
        ].values[0]

        target_game = df_games.loc[
            df_games['appid'] == user_games[-1], 'name'
        ].values[0]

        recommendations = game_recommender.get_recommendations(
            input_game,
            df_games,
            matrix,
            top_n=max(k_values)
        )['name'].tolist()

        for k in k_values:
            hits[k] += int(target_game in recommendations[:k])

    except:
        continue


# Results
print("\nRESULTS")
for k in k_values:
    print(f"Recall@{k}: {hits[k] / len(valid_users):.0%}")