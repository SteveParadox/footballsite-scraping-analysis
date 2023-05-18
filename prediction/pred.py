from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings


from _ext import prediction, start

def get_fixtures():
    try:
        for_team, against_team, any_win, data_, data = prediction()
        start_date = datetime.today() + timedelta(days=1)
        end_date = start_date + timedelta(days=3)
        delta = timedelta(days=1)
        match_fix = []
        compiled_for = set()
        compiled_against = set()
        compiled_any = set()
        
        while start_date <= end_date:
            date_str = start_date.strftime('%Y-%m-%d')
            url = f'https://www.bbc.com/sport/football/scores-fixtures/{date_str}'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            for fixture in soup.select('.sp-c-fixture'):
                team_1 = fixture.select_one('.sp-c-fixture__team--time-home .qa-full-team-name')
                team_2 = fixture.select_one('.sp-c-fixture__team--time-away .qa-full-team-name')

                if team_1 is not None and team_2 is not None:
                    team_1 = fixture.select_one('.sp-c-fixture__team--time-home .qa-full-team-name').text
                    team_2 = fixture.select_one('.sp-c-fixture__team--time-away .qa-full-team-name').text
                    match_time = fixture.select_one('.sp-c-fixture__number--time').text
                    match_fix.append(f'{team_1} vs {team_2}')
                else:
                    pass

            start_date += delta
           
        selection_set = set(line.strip() for line in for_team)
        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_for.add(item)
                
        selection_set = set(line.strip() for line in against_team)
        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_against.add(item)

        selection_set = set(line.strip() for line in any_win)
        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_any.add(item)

        return compiled_for, compiled_against, compiled_any, match_fix, data_, for_team, against_team, any_win, data

    except Exception as e:
        print(f"Error: {e}")
        return None 

def arrange_fixture():
    # Removing unwanted data and organizing the fixtures into sections
    compiled_for, compiled_against, compiled_any, match_fix, data_, for_team, against_team, any_win, data = get_fixtures()
    compiled_for = {match for match in compiled_for if 'Ladies' not in match and 'Women' not in match }
    compiled_against = {match for match in compiled_against if 'Ladies' not in match and 'Women' not in match }
    compiled_any = {match for match in compiled_any if 'Ladies' not in match and 'Women' not in match }

    return compiled_for, compiled_against, compiled_any, match_fix, data_, for_team, against_team, any_win, data


def win_fav():
    compiled_for, compiled_against, compiled_any, match_fix, data_, for_team, against_team, any_win, data = arrange_fixture() # Cleaned Fixtures

    compiled_for = [[s.strip() for s in item.split('vs')] for item in compiled_for]
    compiled_against = [[s.strip() for s in item.split('vs')] for item in compiled_against]
    compiled_any = [[s.strip() for s in item.split('vs')] for item in compiled_any]

    comp_ = []
    _comp = []
    pick_for = []
    pick_against = []

    for pair in compiled_against:
        sublist = []
        for item in pair:
            for row in data:
                if item in row:
                    sublist.extend(row)
        if len(sublist) == 26:
            comp_.append(sublist)
       
    for pair in compiled_for:
        sublist = []
        for item in pair:
            for row in data:
                if item in row:
                    sublist.extend(row)
        if len(sublist) == 26:
            _comp.append(sublist)

    for i in comp_: 
        if i[-1] >= 2 and i[15] >= 14:
            pick_against.append(i[0])
        elif i[12] >= 2 and i[2] >= 14: 
            pick_against.append(i[13])
    for i in _comp:
        if i[-1] <= -1 and i[15] <= 13:
            pick_for.append(i[0])
        elif i[12] <= -1 and i[2] <= 13:
            pick_for.append(i[13])
    
    return pick_for, pick_against, match_fix, data_


def plus_goals():

    _, _, match_fix, data_ = win_fav()

    # Spliting fixtures into home team and away team
    match_fix = [[s.strip() for s in item.split('vs')] for item in match_fix]
    home_teams = [item[0] for item in match_fix]
    away_teams = [item[1] for item in match_fix]

# structuring team table with pandas
    df = pd.DataFrame(data_, columns=["Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points",
                                     "Last_5_W", "Last_5_D", "Last_5_L", "Team_Form", "Outcome"])

    df.loc[:, 'Home_Team'] = home_teams * (len(df) // len(home_teams)) + home_teams[:len(df) % len(home_teams)]
    df.loc[:, 'Away_Team'] = away_teams * (len(df) // len(away_teams)) + away_teams[:len(df) % len(away_teams)]


    # Data preprocessing to convert team names to numbers
    label_encoder = LabelEncoder()
    df['team_encoded'] = label_encoder.fit_transform(df['Team'])

    # Feature engineering
    df['avg_goals_scored'] = df['GF'] / df['Played']
    df['recent_form'] = df.groupby('Team')['GF'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['avg_goals_scored_opponent'] = df.groupby('Away_Team')['GA'].transform('mean')
    df['avg_goals_conceded_opponent'] = df.groupby('Home_Team')['GA'].transform('mean')

    # Splitting into training and testing sets
    features = ['team_encoded', 'avg_goals_scored', 'recent_form', 'avg_goals_scored_opponent', 'avg_goals_conceded_opponent']
    target = 'GF'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiating and training the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluating the model on the testing set
    score = model.score(X_test, y_test)
    print("Model Score:", score)

    # trained model
    return model, df, match_fix

def predict_goals():
    model, df, match_fix = plus_goals()

    # features for a team to predict the likelihood of scoring up to two goals
    team_features = [[df['team_encoded'].values[0], df['avg_goals_scored'].values[0], df['recent_form'].values[0],
                      df['avg_goals_scored_opponent'].values[0], df['avg_goals_conceded_opponent'].values[0]]]
    predicted_goals = model.predict(team_features)

    # Predicting the number of goals for each fixture
    fixture_predictions = []

    for i in range(len(match_fix)):
        team_features = [[df['team_encoded'].values[i], df['avg_goals_scored'].values[i], df['recent_form'].values[i],
                          df['avg_goals_scored_opponent'].values[i], df['avg_goals_conceded_opponent'].values[i]]]
        predicted_goals = model.predict(team_features)
        fixture_predictions.append(predicted_goals[0])

    # fixtures with two or more goals
    _more_goals = [match_fix[i] for i, _prediction in enumerate(fixture_predictions) if _prediction >= 50]
    print("Fixtures with Two or More Goals:", _more_goals)

    return df, match_fix

    
def statistical_analysiis():
    df, match_fix = predict_goals()

    for fixture in match_fix:
        home_team = fixture[0]
        away_team = fixture[1]

        home_team_info = df[df["Team"] == home_team]
        away_team_info = df[df["Team"] == away_team]

        if not home_team_info.empty and not away_team_info.empty:
            
            home_team_played = home_team_info["Played"].values[0]
            home_team_win = home_team_info["Won"].values[0]
            home_team_draw = home_team_info["Drawn"].values[0]
            home_team_loss = home_team_info["Lost"].values[0]
            home_team_form = home_team_info["Team_Form"].values[0]
            home_team_goals_for = home_team_info["GF"].values[0]
            home_team_goals_against = home_team_info["GA"].values[0]
            home_avg_goals = (home_team_goals_for / home_team_goals_against) * home_team_form
            
            away_team_played = away_team_info["Played"].values[0]
            away_team_win = away_team_info["Won"].values[0]
            away_team_draw = away_team_info["Drawn"].values[0]
            away_team_loss = away_team_info["Lost"].values[0]
            away_team_form = away_team_info["Team_Form"].values[0]
            away_team_goals_for = away_team_info["GF"].values[0]
            away_team_goals_against = away_team_info["GA"].values[0]
            away_avg_goals = (away_team_goals_for / away_team_goals_against) * away_team_form

            total_goals = home_avg_goals + away_avg_goals
            

            # Set the threshold for the number of goals
            one_threshold = 2
            two_threshold = 3 
            three_threshold = 4

            if total_goals > one_threshold:
                #print("More than Zero goal")
                goal_prediction =  fixture
                #print(goal_prediction)
        
            if total_goals > two_threshold:
                #print("More than one goal")
                goal_prediction = fixture
                print(goal_prediction)
                
            if total_goals > three_threshold:
                #print("More than two goal" )
                goal_prediction = fixture
                #print(goal_prediction)
            


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    pick_for, pick_against, match_fix, data_ = win_fav()
    
    print(pick_for)
    print(pick_against)
    print(statistical_analysiis())
    end= time.time()
    print(end - start)
    
