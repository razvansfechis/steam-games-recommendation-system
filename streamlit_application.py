import streamlit
import pandas as pd
import base64
import game_recommender

# Set the page's title, the page's layout and the tab's title
streamlit.set_page_config(page_title="Steam Games Recommender System", layout="wide")
streamlit.title("Steam Games Recommender System")

# Convert the image file into a text string that will be decoded and injected into the HTML
# because Streamlit can't add backgrounds by itself
with open("background.jpg", "rb") as file:
    img_string = base64.b64encode(file.read()).decode()

# inject the decoded background  directly into the app background
# stApp is the main background wrapper used by Streamlit
# use unsafe_allow_html to use raw HTML on the Streamlit
streamlit.markdown(
    f"""
    <style>
    .stApp {{ 
        background-image: url("data:image/jpg;base64,{img_string}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the data ONCE when the program is firstly run (this is what @streamlit.cache_resource does)
@streamlit.cache_resource(show_spinner="Loading the data, please wait ...")
def load_data():
    df = pd.read_csv("data/13_steam_games_final.csv")
    return game_recommender.build_recommendation_system(df)

tfidf_matrix, df = load_data()

# Get the list of games
game_list = df['name'].astype(str).unique()

# Pick the game you like
selected_game = streamlit.selectbox("Choose a game you like:", options=game_list)

# After pressing the Recommend Games button
if streamlit.button("Recommend Games", type="primary"):

    # Get the recommendations
    recommendations = game_recommender.get_recommendations(selected_game, df, tfidf_matrix, top_n=5)

    streamlit.markdown(f"### You might like these similar games:")

    # Parse each recommended game in a for loop
    # games_rank is an index, starting and 1 and (_, row) is the data
    for games_rank, (_, row_data) in enumerate(recommendations.iterrows(), 1):
        # Use a container for each game
        with streamlit.container(border=True):
            # Split the data (header image -> image_column and game name and description -> name_and_description_column)
            # into parts with image_column being 25% of the total width and name_and_description_column
            # being 75% of the total width
            image_column, name_and_description_column = streamlit.columns([1, 3])

            with image_column:
                # Checks if there is a link for an image and if the image has a valid value (not only spaces)
                if pd.notna(row_data['header_image_url']) and str(row_data['header_image_url']).strip() != "":
                    streamlit.image(row_data['header_image_url'], width="stretch")
                else:
                    streamlit.caption("No Image Available")

            with name_and_description_column:
                # Display the game's rank in the recommendations and it's name
                streamlit.markdown(f"### **{games_rank}. {row_data['name']}**")

                # Displays the short description right below the title, if there exists a short description and has
                # a valid value (not only spaces)
                if pd.notna(row_data['short_description']) and str(row_data['short_description']).strip() != "":
                    streamlit.write(row_data['short_description'])
                else:
                    streamlit.caption("*No description available for this game.*")