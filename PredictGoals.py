import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


def get_result_label(row):
    if row['home_goals'] > row['away_goals']:
        return 0 
    elif row['home_goals'] == row['away_goals']:
        return 1  
    else:
        return 2  
def load_and_prepare_data():
    data = pd.read_csv("all_seasons.csv")
    data['result_label'] = data.apply(get_result_label, axis=1)
    features = [
        'home_shots_on_target', 'home_shots', 'home_fouls', 'home_corners',
        'home_offsides', 'home_possession', 'home_yellow_cards', 'home_red_cards',
        'home_goalkeeper_saves', 'home_attempted_passes', 'home_successful_passes',
        'away_shots_on_target', 'away_shots', 'away_fouls', 'away_corners',
        'away_offsides', 'away_possession', 'away_yellow_cards', 'away_red_cards',
        'away_goalkeeper_saves', 'away_attempted_passes', 'away_successful_passes'
    ]
    
    data = data.dropna(subset=features + ['result_label', 'home_goals', 'away_goals'])
    
    X = data[features]
    y_cls = data['result_label']
    y_home_goals = data['home_goals']
    y_away_goals = data['away_goals']
    return data,features ,X, y_cls, y_home_goals, y_away_goals

def train_model(X, y_cls, y_home_goals, y_away_goals):
    clf = RandomForestClassifier()
    clf.fit(X, y_cls)
    
    home_reg = GradientBoostingRegressor()
    home_reg.fit(X, y_home_goals)
    
    away_reg = GradientBoostingRegressor()
    away_reg.fit(X, y_away_goals)

    return clf , home_reg , away_reg

def predict(X ,clf ,home_reg ,away_reg):
    y_cls_pred = clf.predict(X)
    classification_accuracy = accuracy_score(y_cls, y_cls_pred)
    
    home_goals_pred = home_reg.predict(X)
    away_goals_pred = away_reg.predict(X)

    home_goals_rmse = np.sqrt(mean_squared_error(y_home_goals, home_goals_pred))
    away_goals_rmse = np.sqrt(mean_squared_error(y_away_goals, away_goals_pred))
    return classification_accuracy , home_goals_rmse , away_goals_rmse

def get_team_stats(data,features,team, is_home, N=5):
    if is_home:
        team_games = data[data['home_team'] == team].sort_values('date', ascending=False).head(N)
        cols = [col for col in features if col.startswith('home_')]
    else:
        team_games = data[data['away_team'] == team].sort_values('date', ascending=False).head(N)
        cols = [col for col in features if col.startswith('away_')]

    return team_games[cols].mean(numeric_only=True)

def predict_inputs(data,features,clf ,home_reg,away_reg,home_team,away_team):
    # home_team = input("Enter the home team name: ").strip()
    # away_team = input("Enter the away team name: ").strip()
    
    home_stats = get_team_stats(data,features,home_team, is_home=True)
    away_stats = get_team_stats(data, features,away_team, is_home=False)
    
    if home_stats.isnull().any() or away_stats.isnull().any():
        print("Not enough data for both teams. Please make sure the team names are correct.")
        return
    else:
        input_row = pd.DataFrame([pd.concat([home_stats, away_stats])])[features]
    
        prediction_cls = clf.predict(input_row)[0]
        pred_home_goals = home_reg.predict(input_row)[0]
        pred_away_goals = away_reg.predict(input_row)[0]
    
        result_map = {
            0: "Home team wins",
            1: "Draw",
            2: "Away team wins"
        }
        
    return pred_home_goals , pred_away_goals
def Calling(home_team,away_team):
    data,features ,X, y_cls, y_home_goals, y_away_goals = load_and_prepare_data()
    clf , home_reg , away_reg = train_model(X, y_cls, y_home_goals, y_away_goals)
    pred_home_goals , pred_away_goals = predict_inputs(data,features,clf ,home_reg,away_reg,home_team,away_team)
    return pred_home_goals , pred_away_goals
    