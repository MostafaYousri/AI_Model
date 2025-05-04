from flask import Flask , request, jsonify

import pandas as pd
import numpy as np

#import PredictMatchWinner
import predictor
import PredictGoals
app = Flask(__name__)

@app.route('/predict/<home_team>/<away_team>', methods=["GET"])
def predict(home_team,away_team):
    proba =  predictor.predictor(home_team, away_team)
    homeGoals , awayGoals = PredictGoals.Calling(home_team,away_team)
    return jsonify ({"Home_Team":home_team,"Away_Team":away_team,"HomeWin":proba[2],"AwayWin":proba[0],"Draw":proba[1],
                     "HomeGoals":homeGoals,"AwayGoals":awayGoals})

if __name__=="__main__":
    app.run(port=5000)