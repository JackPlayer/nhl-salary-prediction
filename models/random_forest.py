import sys
sys.path.append("..")
from data import data_loader
from utils import feature_list

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

player_data = data_loader.load_player_data('../data/player_dataset.csv')

model = RandomForestRegressor(random_state = 1)

y = player_data['SALARY']
X = player_data[feature_list]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

model.fit(train_X, train_y)

predict_y = model.predict(val_X)

mse =  mean_absolute_error(val_y, predict_y)

print("MSE (Random Forest): ", mse)