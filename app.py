import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.metrics import accuracy_score
from flask import Flask, url_for, session, request, redirect, render_template
import random


app = Flask(__name__)

# NOTES: Line 1-56 is all exploration and was initially done on jupyter notebooks but I cant find that so I just put everything here

# Load betting data that has been scrapped
betting_db = pd.read_csv('betting_db.csv')

# All betting providers with home and away odds
all_bets = ['B365', 'B365', 'B365', 'BW', 'BW', 'BW', 'IW', 'IW', 'IW', 'PS', 'PS', 'PS', 'WH', 'WH', 'WH', 'VC', 'VC', 'VC']

# Football logos for all clubs available in the product
pic_dict = {'Burnley':'Burnley_F.C._Logo.svg.png', 'Wolves':'Wolverhampton_Wanderers.svg.png', 'Brentford':'Brentford_FC_crest.svg.png',
            'Watford':'Watford.svg.png', 'Leicester City':'Leicester_City_crest.svg.png', 'Aston Villa':'Aston_Villa_FC_crest_(2016).svg.png',
            'Crystal Palace': 'Crystal_Palace_FC_logo_(2022).svg.png', 'Brighton':'Brighton_&_Hove_Albion_logo.svg.png',
            'Southampton':'FC_Southampton.svg.png', 'Chelsea':'Chelsea_FC.svg.png', 'Manchester Utd':'Manchester_United_FC_crest.svg.png',
            'Everton': 'Everton_FC_logo.svg.png', 'Leeds United':'Leeds_United_F.C._logo.svg.png', 'Newcastle Utd':'Newcastle_United_Logo.svg.png', 'West Ham':'West_Ham_United_FC_logo.svg.png', 'Arsenal':'Arsenal-Logo.png', 'Tottenham':'Tottenham_Hotspur.svg.png',
 'Manchester City':'Manchester_City_FC_badge.svg.png'}


# load data for current standings to make decisions when current model can't
current_table = pd.read_csv('Book2.csv')

# load performance data and apply udpates to categorical data
# clean up so that data be machine readable
matches = pd.read_csv('matches.csv', index_col=0)
matches['date'] = pd.to_datetime(matches['date'])
matches["target"] = matches["result"].astype('category').cat.codes
matches['venue_code'] = matches['venue'].astype('category').cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek

# apply random forrest classifier and divide data into train and test
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] > '2022-01-01']

# Define predictor variable, train and calculate accuracy
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])
error = accuracy_score(test["target"], preds)

# create crosstab to check preformance
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
test_df = pd.crosstab(index=combined['actual'], columns=combined['predicted'])
prec_score = precision_score(test['target'], preds, average='weighted')


# get rolling averages to account for recent form to improve accuracy
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# create new columns for rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# apply rolling_averages function to all teams
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

# drop team name index level to make the df easier to work with
matches_rolling = matches_rolling.droplevel('team')

# function to make prediction again with test and train data set
def make_predictions(data, predictors):
    data.to_csv('testing_data.csv')
    # Train dataset
    train = data[data["date"] < '2022-01-01']
    # Test dataset
    test = data[data["date"] > '2022-01-01']
    # train random forest regression on predictors and training dataset
    rf.fit(train[predictors], train["target"])
    # test predictions on test dataset
    preds = rf.predict(test[predictors])
    # get data frame of predictions
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    # get precision value (~50-60%)
    error = precision_score(test["target"], preds, average='micro')
    return combined, error

# get combined_df and precision
combined, error = make_predictions(matches_rolling, predictors + new_cols)

# get dataframe to see where the prediction is going wrong, explore where we are seeing issues
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined)
# There are discrepencies between team names (wolves vs wolverhamption wanderes)

# create class that inherits from dict class to avoid map method removing missing keys
class MissingDict(dict):
    __missing__ = lambda self, key: key

# create dict to map values for normalization
map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"}
# map inconsistent names to std names we will use
mapping = MissingDict(**map_values)
# Create new columnd with std names
combined["new_team"] = combined["team"].map(mapping)

# loading old combined object to get predictions for later
new_combined = pd.read_csv('final_explore.csv')

# combine with merge df
# can be used to calculate final accuracy
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

# get predictions from our test dataset
# this should only get historic predictions of matches between two teams ignoring home/away
def get_prediction(team1, team2):
    row_needed = list(new_combined.loc[(new_combined['team'] == team1) & (new_combined['opponent'] == team2)]['predicted'])
    row_needed2 = list(new_combined.loc[(new_combined['team'] == team2) & (new_combined['opponent'] == team1)]['predicted'])
    # look at home/away permutation if it exsists
    if len(row_needed) != 0:
        print(row_needed[0])
        return row_needed[0]
    # look at away/home permutation if it exsits
    elif len(row_needed2) != 0:
        return row_needed2[0]
    # if neither exist, make predictions based on JUST the standings in the table, account for missing data, eventually will stop using this
    else:
        team1_pos = current_table.loc[current_table['Team'] == team1, 'Position'].iloc[0]
        team2_pos = current_table.loc[current_table['Team'] == team2, 'Position'].iloc[0]
        if team2_pos > team1_pos:
            return 2
        else:
            return 1

# get list of teams it will function with, because there are missing teams in our data, we don't want to let users make predictions for anyone
allowed_list = list(new_combined['team'].unique())

# missing data for Norwich so drop that (sorry, but its a PoC so don't hate m8)
allowed_list.remove('Norwich City')

# FlaskApp #CoolPeopleUseFlask #IUsedBothCamelCase_and_snake_Case #Quirky #single

# Create homepage
@app.route('/', methods=['GET', 'POST'])
def homepage():
    # get what the user inputs as teams
    team1 = request.args.get('exampleInputEmail1')
    team2 = request.args.get('exampleInputEmail12')
    # randomize how the teams appear on the FE just for fun
    allowed_list1 = random.sample(allowed_list, len(allowed_list))
    # Check to see if the teams have actually been entered
    # Ideally this should be done via JS on the FE
    if team1 is not None:
        # get the results from our testing dataset
        result_all = get_final_result(team1, team2)
        # See if we are actually getting a result, if its a draw that is predicted then we won't have a hedged bet
        if len(result_all) == 4:
            # look at function below
            # predicted results
            result = result_all[0]
            # 70% bet, odds to match our predictions
            active_bet = result_all[1]
            # 30$ bet, odds to go against our prediction
            hedge_bet = result_all[2]
            # precision for prediction
            error = result_all[3]
            # final betting strategy table
            # This can also be constructed on the front end with teh infromation above but was lazy
            bet_strat = get_bet(active_bet, hedge_bet, error)
            # render page with all information
            return render_template('/home.html', allowed_list=allowed_list, allowed_list1=allowed_list1, team1=team1, team2=team2, resultText=result,
                                   img1=pic_dict[team1], img2=pic_dict[team2], active_bet=active_bet,
                                   hedge_bet=hedge_bet, error=error, tables=[bet_strat.to_html()])

        else:
            # same as above but handles case for draw
            # in a draw we will ONLY have an active_bet and not hedge our bets
            # more complex handling of draws should be done as they are likely fairly common
            result = result_all[0]
            active_bet = result_all[1]
            error = result_all[2]
            bet_strat = pd.DataFrame({'active_allocation': 100, 'confidence': error, 'EV': 1*active_bet}, index=[0])
            return render_template('/home.html', allowed_list=allowed_list, allowed_list1=allowed_list1, team1=team1, team2=team2, resultText=result,
                                   img1=pic_dict[team1], img2=pic_dict[team2], active_bet=active_bet, error=error,tables=[bet_strat.to_html()])
    # Page gets displayed even without user input data
    return render_template('home.html', allowed_list=allowed_list, allowed_list1=allowed_list1)

def get_bet(active_bet, hedge_bet, error):
    if active_bet > 10*hedge_bet:
        bet_strategy = {'active_allocation': 100, 'confidence': error, 'best odds':hedge_bet, 'EV': (hedge_bet * error/100)}
        bet_strategy = pd.DataFrame(bet_strategy, index=[0])
        return bet_strategy
    else:
        odd2 = active_bet
        odd1 = hedge_bet
        ev1 = (odd1) * 0.7 * (error / 100)
        ev2 = (odd2) * 0.3 * ((100-error) / 100)
        bet_strategy = [{'active allocation': 70, 'confidence':error, 'best odds':odd1, 'EV': ev1}, {'active allocation': 30, 'confidence':(100-error), 'best odds':odd2, 'EV': -ev2},
                        {'active allocation': 100, 'confidence':'NA', 'best odds': 'NA', 'EV':(ev1 - ev2)}]
        bet_strategy = pd.DataFrame(bet_strategy)
        return bet_strategy

# get final results of the matchup between the two teams
# get error values
# get prediction
# package all up for sending to front end
def get_final_result(team1,team2):
    # get the predictions of the outcomes
    final_result = get_prediction(team1, team2)
    # initialize variables so can be used in further computation (WIP for future release, can just be returned in the if statements)
    odds_list = []
    odds_list_win = []
    odds_list_lose = []
    max_bet_draw = 0
    max_bet_win = 0
    max_bet_lost = 0
    # If Home team wins get the accuracy for that prediction
    if final_result == 2:
        need_row1 = betting_db[(betting_db['HomeTeam'] == team1) & (betting_db['AwayTeam'] == team2)]
        error1 = need_row1['error'].iloc[0]
    # If Away team wins get the accuracy
    else:
        need_row1 = betting_db[(betting_db['HomeTeam'] == team2) & (betting_db['AwayTeam'] == team1)]
        error1 = need_row1['error'].iloc[0]
    # if its a draw get the best odds for that team to win and pass error value for FE
    if final_result == 0:
        for x in all_bets:
            odds_list.append(need_row1[f'{x}D'].iloc[0])
            max_bet_draw = max(odds_list)
        return f'Its going to be a draw we are {round(error1)}% sure', max_bet_draw, error1
    # if away team wins get the best odds for that team to win and pass error value for FE
    elif final_result == 1:
        for x in all_bets:
            # get odds of away team winning and get the value from that row (active bet)
            a = betting_db.loc[(betting_db['HomeTeam'] == team2) & (betting_db['AwayTeam'] == team1), f'{x}A'].iloc[0]
            # create a list of betting odds for this outcome
            odds_list_win.append(a)
            # get the odds with the highest payout
            max_bet_win = max(odds_list_win)
            # get the odds of the away team loosing
            y = betting_db.loc[(betting_db['HomeTeam'] == team1) & (betting_db['AwayTeam'] == team2), f'{x}H'].iloc[0]
            # create a list of all the ods
            odds_list_lose.append(y)
            # get the highest payout
            max_bet_lost = max(odds_list_lose)
        return f'{team2} are probably going to win! we are {round(error1)}% sure', max_bet_win, max_bet_lost, error1
    # if home team wins get the best odds for that team to win and pass error value for FE
    else:
        # identical to above but for home team to win
        for x in all_bets:
            a = betting_db.loc[(betting_db['HomeTeam'] == team1) & (betting_db['AwayTeam'] == team2), f'{x}A'].iloc[0]
            odds_list_win.append(a)
            max_bet_win = max(odds_list_win)
            y = betting_db.loc[(betting_db['HomeTeam'] == team1) & (betting_db['AwayTeam'] == team2), f'{x}H'].iloc[0]
            odds_list_lose.append(y)
            max_bet_lost = max(odds_list_lose)
        return f'{team1} are probably going to win! we are {round(error1)}% sure', max_bet_win, max_bet_lost, error1

if __name__ == "__main__":
    app.run()
