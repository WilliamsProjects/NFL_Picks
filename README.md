# NFL_Picks
Predicting the outcome of NFL games in Python using the scikit-learn package

## Overview

I'm really interested in the NFL and have predicted every game for the past 2 years. I wanted to make a model which could out-pick me. Whats more, I hadn't done a large machine learning project at university at this point, so was keen learn some new skills!

The goal of the analysis is to use games from the 2017-2019 NFL seasons as training data to build a logistic regression model to pick the 2020 season.

## Methods and outcomes

I used the SportsReference package built into Python to extract NFL game data. I recorded every game (including postseason) from 2017-2020 in a pandas dataframe, specifically the number of 1st downs gained and allowed by the home/away teams and the number of total yards gained and allowed by the home/away teams. I picked these statistics because it gives an indication of a team's quality both offensively and defensively and are good predictors for team success.

I used this dataframe to find the average number of 1st downs/yards gained/allowed (4 separate statistics) up to that point in the season. For example, in the 57th match of the 2017 season the Dallas Cowboys faced the LA Rams. The Cowboys averaged 17.9 1st downs and 311 yards gained per game up to that point in the 2017 season. These features are used to predict who will win a given matchup.

A logistic regression model was used to predict whether the home team for a given matchup would win (output = 1) or lose (output = 0). The data was standardised to decrease the variance in data points.

The model achieved an accuracy of 68% for the 2020 season. For reference, the fivethirtyeight.com model had an accuracy of 69% and Gridiron AI had an accuracy of 69.4%.  As my (more simple) model achieves an accuracy similar to these highly regarded publicly available models, the model seems to be on the right track. I also picked at 68% accuracy for the 2020 season (what a coincidence!)

I was also interested to see how much money I would win (or lose) if I bet $100 on every game where my model predicted a significantly different likelihood of victory for one team than the betting odds did. For example, if my model said that the New England Patriots had a 75% chance of winning, but the betting odds said the Patriots only had a 65% chance of winning, I would bet on the Patriots. 

Given I bet on every game in the 2020 season where the difference in expected win probability between my model and the betting odds exceeded 7.5%, I would have lost $124 (1.24 units). Only 24 games out of 269 ( in the 2020 season had a win probability difference between my model and the betting odds of greater than 7.5%. As betting odds are very good indicators of the winning probability and my model mostly stayed within 7.5% of these probabilities, my model's predictions are within reason.

## Potential drawbacks in the model

The main improvement to be made to my model is to use cross validation. Because of the clear test/train split (using 2017-2019 to predict 2020), I didn't use cross validation. This is because I didn't want to use data from 2020 games to predict games from 2020. 

I aim to improve the model by examining more closely which features should be kept, which should be dropped and which should be introduced to the model. I will test this model on the upcoming 2021 season.

## Key learnings

My aim was to learn how to do machine learning in Python, so I familiarised myself with the sci-kit learn package. I also became more familiar with Pandas. Furthermore, I learnt about cross validation and the standardisation of data to improve machine learning models.


