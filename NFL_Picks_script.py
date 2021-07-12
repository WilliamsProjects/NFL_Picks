from sportsreference.nfl.boxscore import Boxscores, Boxscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from NFL_Picks_functions import *


#Reading in data from sportreference. This takes ages so I saved it the data to text files and went from there

if False:
    lines_2017 = pd.read_csv('2017_lines.csv')
    lines_2018 = pd.read_csv('2018_lines.csv') 
    lines_2019 = pd.read_csv('2019_lines.csv')
    lines_2020 = pd.read_csv('2020_lines.csv')



    games_df_2020 = pd.DataFrame(columns=['Home_Team', 'Away_Team', '1D_gain_H', '1D_gain_A', '1D_allowed_H',
    '1D_allowed_A', 'Yards_gain_H', 'Yards_gain_A', 'Yards_allowed_H', 'Yards_allowed_A', 'Winner']) #Making new dataframe for 2020 data


    for i in range(1,22):
        
        for j in range(len(Boxscores(i,2020).games[str(i)+'-2020'])):
            game_str = Boxscores(i,2020).games[str(i)+'-2020'][j]['boxscore']
            game_stats = Boxscore(game_str)
            games_df_2020 = games_df_2020.append({'Home_Team' : Boxscores(i,2020)._boxscores[str(i)+'-2020'][j]['home_name'],
            'Away_Team' : Boxscores(i,2020)._boxscores[str(i)+'-2020'][j]['away_name'], '1D_gain_H' : game_stats.home_first_downs,
            '1D_gain_A' : game_stats.away_first_downs, '1D_allowed_H' : game_stats.away_first_downs,
            '1D_allowed_A' : game_stats.home_first_downs, 'Yards_gain_H': game_stats.home_total_yards, 
            'Yards_gain_A': game_stats.away_total_yards, 'Yards_allowed_H': game_stats.away_total_yards,
            'Yards_allowed_A': game_stats.home_total_yards, 'Winner' :  game_stats.winner }, 
                    ignore_index = True)
            games_df_2020.to_csv('2020_games.csv',index=False)
            


    games_df_201x = pd.DataFrame(columns=['Home_Team', 'Away_Team', '1D_gain_H', '1D_gain_A', '1D_allowed_H',
    '1D_allowed_A', 'Yards_gain_H', 'Yards_gain_A', 'Yards_allowed_H', 'Yards_allowed_A', 'Winner']) #Using 1st downs and yards gained and allowed

    years = [2017, 2018, 2019] #Using previous 3 years to predict 2020 season
    for h in range(len(years)):       
        for i in range(1,22):
            for j in range(len(Boxscores(i,years[h]).games[str(i)+'-'+str(years[h])])):
                game_str = Boxscores(i,years[h]).games[str(i)+'-'+str(years[h])][j]['boxscore']
                game_stats = Boxscore(game_str)
                games_df_201x = games_df_201x.append({'Home_Team' : Boxscores(i,years[h])._boxscores[str(i)+'-'+str(years[h])][j]['home_name'],
                'Away_Team' : Boxscores(i,years[h])._boxscores[str(i)+'-'+str(years[h])][j]['away_name'], '1D_gain_H' : game_stats.home_first_downs,
                '1D_gain_A' : game_stats.away_first_downs, '1D_allowed_H' : game_stats.away_first_downs,
                '1D_allowed_A' : game_stats.home_first_downs, 'Yards_gain_H': game_stats.home_total_yards, 
                'Yards_gain_A': game_stats.away_total_yards, 'Yards_allowed_H': game_stats.away_total_yards,
                'Yards_allowed_A': game_stats.home_total_yards, 'Winner' : game_stats.winner }, 
                        ignore_index = True)
                games_df_201x.to_csv('2017-2019_games.csv',index=False)

#2020 stats (test data)

Stats_2020 = pd.read_csv("2020_games.csv")
Stats_2017 = pd.read_csv("2017-2019_games.csv")  

Stats_2020 = Stats_2020.drop(['Unnamed: 12', 'Concat1', 'Concat2', 'Line.1'],axis=1) #Drop unused columns from csv files
Stats_2017 = Stats_2017.drop(['Unnamed: 12', 'Concat1', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'],axis=1)

home_team = Stats_2020["Home_Team"]
away_team = Stats_2020["Away_Team"]

#Finding the average 1st downs and yards gained and allowed by the home and away teams

Stats_2020["Sum_1D_gain_H"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_1D_gain_A"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_1D_allowed_H"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_1D_allowed_A"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_Yards_gain_H"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_Yards_gain_A"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_Yards_allowed_H"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])
Stats_2020["Sum_Yards_allowed_A"] = pd.Series([0.0 for x in range(Stats_2020.shape[0])])

for i in range(Stats_2020.shape[0]):

    #Find games played by home team and away team in a specific game 
    home_team_index1 = home_team[home_team == home_team[i]].index
    home_team_index2 = home_team[away_team == home_team[i]].index #home now, away before
    away_team_index1 = away_team[away_team == away_team[i]].index
    away_team_index2 = away_team[home_team == away_team[i]].index #away now, home before

    #Find games played up to this point in season by home team and away team in a specific game 
    homehome = [home for home in home_team_index1 if home < i]
    homeaway = [home for home in home_team_index2 if home < i]
    awayhome = [away for away in away_team_index2 if away < i]
    awayaway = [away for away in away_team_index1 if away < i]

    #Find num games played by home team and away team
    num_games_H = len(homehome) + len(homeaway)
    num_games_A = len(awayhome) + len(awayaway)

    #Up to this point in season, get average 1D and yards gained/allowed by home and away team for that game
    try:
        Stats_2020["Sum_1D_gain_H"][i] = (sum(np.array(Stats_2020["1D_gain_H"][homehome])) + sum(np.array(Stats_2020["1D_gain_A"][homeaway])))/num_games_H
        Stats_2020["Sum_1D_gain_A"][i] = (sum(np.array(Stats_2020["1D_gain_H"][awayhome])) + sum(np.array(Stats_2020["1D_gain_A"][awayaway])))/num_games_A

        Stats_2020["Sum_1D_allowed_H"][i] = (sum(np.array(Stats_2020["1D_allowed_H"][homehome])) + sum(np.array(Stats_2020["1D_allowed_A"][homeaway])))/num_games_H
        Stats_2020["Sum_1D_allowed_A"][i] = (sum(np.array(Stats_2020["1D_allowed_H"][awayhome])) + sum(np.array(Stats_2020["1D_allowed_A"][awayaway])))/num_games_A

        Stats_2020["Sum_Yards_gain_H"][i] = (sum(np.array(Stats_2020["Yards_gain_H"][homehome])) + sum(np.array(Stats_2020["Yards_gain_A"][homeaway])))/num_games_H
        Stats_2020["Sum_Yards_gain_A"][i] = (sum(np.array(Stats_2020["Yards_gain_H"][awayhome])) + sum(np.array(Stats_2020["Yards_gain_A"][awayaway])))/num_games_A

        Stats_2020["Sum_Yards_allowed_H"][i] = (sum(np.array(Stats_2020["Yards_allowed_H"][homehome])) + sum(np.array(Stats_2020["Yards_allowed_A"][homeaway])))/num_games_H
        Stats_2020["Sum_Yards_allowed_A"][i] = (sum(np.array(Stats_2020["Yards_allowed_H"][awayhome])) + sum(np.array(Stats_2020["Yards_allowed_A"][awayaway])))/num_games_A

    except:
        Stats_2020["Sum_1D_gain_H"][i] = 0
        Stats_2020["Sum_1D_gain_A"][i] = 0
        Stats_2020["Sum_1D_allowed_H"][i] = 0
        Stats_2020["Sum_1D_allowed_A"][i] = 0

        Stats_2020["Sum_Yards_gain_H"][i] = 0
        Stats_2020["Sum_Yards_gain_A"][i] = 0

        Stats_2020["Sum_Yards_allowed_H"][i] = 0
        Stats_2020["Sum_Yards_allowed_A"][i] = 0

Stats_2020["Winner"][Stats_2020["Winner"] == "Home"] = 1
Stats_2020["Winner"][Stats_2020["Winner"] == "Away"] = 0


#2017-2019 stats (train data)

home_team = Stats_2017["Home_Team"]
away_team = Stats_2017["Away_Team"]

Stats_2017["Sum_1D_gain_H"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_1D_gain_A"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_1D_allowed_H"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_1D_allowed_A"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_Yards_gain_H"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_Yards_gain_A"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_Yards_allowed_H"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])
Stats_2017["Sum_Yards_allowed_A"] = pd.Series([0.0 for x in range(Stats_2017.shape[0])])

adder = 0

#Finding average stats for that specific season up to that game
for cycle in range(3):
    for i in range(int(Stats_2017.shape[0]/3)):

        home_team_index1 = home_team[home_team == home_team[i+adder]].index
        home_team_index2 = home_team[away_team == home_team[i+adder]].index #home now, away before
        away_team_index1 = away_team[away_team == away_team[i+adder]].index
        away_team_index2 = away_team[home_team == away_team[i+adder]].index #away now, home before

        homehome = [home for home in home_team_index1 if(home < adder + i and home >= adder)]
        homeaway = [home for home in home_team_index2 if(home < adder + i and home >= adder)]
        awayhome = [away for away in away_team_index2 if(away < adder + i and away >= adder)]
        awayaway = [away for away in away_team_index1 if(away < adder + i and away >= adder)]

        num_games_H = len(homehome) + len(homeaway)
        num_games_A = len(awayhome) + len(awayaway)


        try:
            Stats_2017["Sum_1D_gain_H"][i+adder] = (sum(np.array(Stats_2017["1D_gain_H"][homehome])) + sum(np.array(Stats_2017["1D_gain_A"][homeaway])))/num_games_H
            Stats_2017["Sum_1D_gain_A"][i+adder] = (sum(np.array(Stats_2017["1D_gain_H"][awayhome])) + sum(np.array(Stats_2017["1D_gain_A"][awayaway])))/num_games_A

            Stats_2017["Sum_1D_allowed_H"][i+adder] = (sum(np.array(Stats_2017["1D_allowed_H"][homehome])) + sum(np.array(Stats_2017["1D_allowed_A"][homeaway])))/num_games_H
            Stats_2017["Sum_1D_allowed_A"][i+adder] = (sum(np.array(Stats_2017["1D_allowed_H"][awayhome])) + sum(np.array(Stats_2017["1D_allowed_A"][awayaway])))/num_games_A

            Stats_2017["Sum_Yards_gain_H"][i+adder] = (sum(np.array(Stats_2017["Yards_gain_H"][homehome])) + sum(np.array(Stats_2017["Yards_gain_A"][homeaway])))/num_games_H
            Stats_2017["Sum_Yards_gain_A"][i+adder] = (sum(np.array(Stats_2017["Yards_gain_H"][awayhome])) + sum(np.array(Stats_2017["Yards_gain_A"][awayaway])))/num_games_A

            Stats_2017["Sum_Yards_allowed_H"][i+adder] = (sum(np.array(Stats_2017["Yards_allowed_H"][homehome])) + sum(np.array(Stats_2017["Yards_allowed_A"][homeaway])))/num_games_H
            Stats_2017["Sum_Yards_allowed_A"][i+adder] = (sum(np.array(Stats_2017["Yards_allowed_H"][awayhome])) + sum(np.array(Stats_2017["Yards_allowed_A"][awayaway])))/num_games_A

        except:
            Stats_2017["Sum_1D_gain_H"][i+adder] = 0
            Stats_2017["Sum_1D_gain_A"][i+adder] = 0
            Stats_2017["Sum_1D_allowed_H"][i+adder] = 0
            Stats_2017["Sum_1D_allowed_A"][i+adder] = 0

            Stats_2017["Sum_Yards_gain_H"][i+adder] = 0
            Stats_2017["Sum_Yards_gain_A"][i+adder] = 0

            Stats_2017["Sum_Yards_allowed_H"][i+adder] = 0
            Stats_2017["Sum_Yards_allowed_A"][i+adder] = 0

        if(i == len(Stats_2017)/3 - 1):
            adder += int(Stats_2017.shape[0]/3)

Stats_2017["Winner"][Stats_2017["Winner"] == "Home"] = 1
Stats_2017["Winner"][Stats_2017["Winner"] == "Away"] = 0

#Only use average data of 1st downs and yards gained/allowed for training/testing of model

X_train = Stats_2017.drop(['Home_Team', 'Away_Team', '1D_gain_H', '1D_gain_A', '1D_allowed_H',
   '1D_allowed_A', 'Yards_gain_H', 'Yards_gain_A', 'Yards_allowed_H', 'Yards_allowed_A', 'Winner'], axis=1)
y_train = Stats_2017["Winner"]

X_test = Stats_2020.drop(['Home_Team', 'Away_Team', '1D_gain_H', '1D_gain_A', '1D_allowed_H',
    '1D_allowed_A', 'Yards_gain_H', 'Yards_gain_A', 'Yards_allowed_H', 'Yards_allowed_A', 'Winner'], axis=1)
y_test = Stats_2020["Winner"]

y_train = y_train.astype('int')

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  #apply scaling on training data
y_test = y_test.astype('int')
print("The accuracy of this model is: {:.3f}".format(pipe.score(X_test, y_test)))

#Make dataframe with hometeam, awayteam, odds of home team winning, odds/prob of hometeam winning by odds, winner

odds_df = pd.DataFrame(columns = ["Home_Team", "Away_Team", "Odds_Home", "Odds_Away", "Prob_Odds_Home", "Prob_Odds_Away", "Prob_Model_Home", "Winner"])

odds_df["Home_Team"] = Stats_2020["Home_Team"]
odds_df["Away_Team"] = Stats_2020["Away_Team"]
odds_df["Winner"] = Stats_2020["Winner"]
odds_df["Prob_Model_Home"] = pipe.predict_proba(X_test)[:,1] #Find probability of home team winning the game


odds_2020 = pd.read_csv("2020_odds.csv")

odds_df["Odds_Home"] = odds_2020["Home_Odds"]
odds_df["Odds_Away"] = odds_2020["Away_Odds"]
odds_df["Decision"] = pd.Series(["Neither" for x in range(odds_df.shape[0])])
odds_df["Units_won_lost"] = pd.Series([0.0 for x in range(odds_df.shape[0])])


count = 0
for i in range(odds_df.shape[0]):
    odds_df["Prob_Odds_Home"][i] = moneyline_prob(odds_df["Odds_Home"][i], odds_df["Odds_Away"][i])
    odds_df["Prob_Odds_Away"][i] = 1 - moneyline_prob(odds_df["Odds_Home"][i], odds_df["Odds_Away"][i])

    if(odds_df["Prob_Model_Home"][i] - odds_df["Prob_Odds_Home"][i] > 0.075):
        odds_df["Decision"][i] = "Home"
        
        if(odds_df["Winner"][i] == 1):
            odds_df["Units_won_lost"][i] = odds_df["Odds_Home"][i] - 1
        else:
             odds_df["Units_won_lost"][i] = -1

    
    #If probability from odds and probability from model for home team differs by more than 7.5%, bet the game

    if(odds_df["Prob_Model_Home"][i] - odds_df["Prob_Odds_Home"][i] < -0.075):
        odds_df["Decision"][i] = "Away"
        
        if(odds_df["Winner"][i] == 0):
            odds_df["Units_won_lost"][i] = odds_df["Odds_Away"][i] - 1
        else:
             odds_df["Units_won_lost"][i] = -1

if(sum(odds_df["Units_won_lost"].values) > 0):
    win = sum(odds_df["Units_won_lost"].values) * 100 
    print("If I bet $100 dollars on all games where the model's probability and the odd's probability for a game differed by 7.5%, I would win ${}".format(win))

if(sum(odds_df["Units_won_lost"].values) < 0):
    loss = abs(sum(odds_df["Units_won_lost"].values)) * 100 
    print("If I bet $100 dollars on all games where the model's probability and the odd's probability for a game differed by 7.5%, I would lose ${}".format(loss))

#Need to update 2017-2019 stats after every week with new week NFL data, then run the model for the next week etc.
#Also, do cross validation on 2020 dataset from 2017-2019 data, maybe add in 2019 to testing data and use 2016 data as well?