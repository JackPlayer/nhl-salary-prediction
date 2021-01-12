import pandas as pd

"""
    Loads the player data from player_dataset.csv and ensures the CAPHIT is a float
"""
def load_player_data(PATH):
    player_data = pd.read_csv(PATH)
    player_data.drop_duplicates(subset=['name'], inplace=True)
    player_data['CAPHIT'] = player_data['CAPHIT'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    return player_data 
