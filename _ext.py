import requests
from bs4 import BeautifulSoup
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
import numpy as np

urls = [
    'https://www.skysports.com/premier-league-table',
    'https://www.skysports.com/la-liga-table',
    'https://www.skysports.com/bundesliga-table',
    'https://www.skysports.com/championship-table',
    'https://www.skysports.com/league-1-table',
    'https://www.skysports.com/scottish-premier-table',
    'https://www.skysports.com/scottish-league-one-table',
    'https://www.skysports.com/scottish-championship-table',
    'https://www.skysports.com/scottish-league-two-table',
    'https://www.skysports.com/serie-a-table',
    'https://www.skysports.com/ligue-1-table',
    'https://www.skysports.com/eredivisie-table',
    'https://www.skysports.com/national-league-table',
    'https://www.skysports.com/national-league-north-table',
    'https://www.skysports.com/national-league-south-table'
]

def teams():
    for_team = []
    against_team = []
    any_win = []
    v = []
    data = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # getting all the tables on the page
        tables = soup.find_all('table')
        for table in tables:
            # Extracting the league name from the table header
            league_name = table.find('caption').text.strip()
            rows = table.find('tbody').find_all('tr')
            last_five_results = []
            for row in rows:
                # Extracting the team name and their statistics
                name = row.find('td', class_='standing-table__cell standing-table__cell--name').text.strip()
                played = int(row.find_all('td')[2].text.strip())
                won = int(row.find_all('td')[3].text.strip())
                drawn = int(row.find_all('td')[4].text.strip())
                lost = int(row.find_all('td')[5].text.strip())
                gf = int(row.find_all('td')[6].text.strip())
                ga = int(row.find_all('td')[7].text.strip())
                gd = int(row.find_all('td')[8].text.strip())
                points = int(row.find_all('td')[9].text.strip())
                td = row.find('td', class_='standing-table__cell is-hidden--bp15 is-hidden--bp35')
             
                if td:
                    form = td.find('div', class_='standing-table__form')
                    if form:
                        spans = form.find_all('span')
                        titles = [span['title'] for span in spans]
                         # Getting the last five results of the team
                        last_five_results = [result.strip() for result in titles[:5]]
                        team_match = ''
                        opp_match = ''
                        # Converting the last five results into a win-draw-loss record
                        last_five_records = []
                        last_five_record = [0, 0, 0]  # Wins, Draws, Losses
                        for result in last_five_results:
                            if name in result:
                                goals = result.split('-')
                                if len(goals) == 2:
                                    if name in goals[0]:
                                        team_match = int(goals[0][-1].strip())
                                        opp_match = int(goals[1][0].strip())
                                    else:
                                        team_match = int(goals[1][0].strip())
                                        opp_match = int(goals[0][-1].strip())
                                    if team_match > opp_match:
                                        last_five_records.append('W')
                                        last_five_record[0] += 1  # Wins
                                    elif team_match == opp_match:
                                        last_five_records.append('D')
                                        last_five_record[1] += 1  # Draws
                                    else:
                                        last_five_records.append('L')
                                        last_five_record[2] += 1  # Losses
                        
                        # Joining the last five records into a string
                        last_five_record_str = '-'.join(last_five_records)
                    # Checking if the team has less than 6 losses or more than 17 losses
                    team_form = last_five_record[0] - last_five_record[2]
                    if played > 20:
                        data.append([name, played, won, drawn, lost, gf, ga, gd, points,
                                 last_five_record[0], last_five_record[1], last_five_record[2], team_form,
                                 ])
                        if lost < 7:
                            for_team.append(name)
                            # Adding the last five records as features for the team
                            if last_five_record_str:
                                feature_name = name.lower().replace(' ', '_') + '_last_5_record'
                                globals()[feature_name] = last_five_record_str 
                        elif lost > 15 or won < 7:
                            against_team.append(name)
                            if last_five_record_str:
                                feature_name = name.lower().replace(' ', '_') + '_last_5_record'
                                globals()[feature_name] = last_five_record_str 
                        elif drawn < 12:
                            any_win.append(name)
                    v.append(last_five_record_str)
                else:
                    print("League is still young .......... ")

    return for_team, against_team, any_win, data

# --------------------------------------------------------------------------------------------------------

def models():
    for_team, against_team, any_win, data = teams()
    data = []  
    # Add the data for each team to the list
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                name = row.find('td', class_='standing-table__cell standing-table__cell--name').text.strip()
                if name in for_team or name in against_team or name in any_win :
                    played = int(row.find_all('td')[2].text.strip())
                    won = int(row.find_all('td')[3].text.strip())
                    drawn = int(row.find_all('td')[4].text.strip())
                    lost = int(row.find_all('td')[5].text.strip())
                    gf = int(row.find_all('td')[6].text.strip())
                    ga = int(row.find_all('td')[7].text.strip())
                    gd = int(row.find_all('td')[8].text.strip())
                    points = int(row.find_all('td')[9].text.strip())
                    outcome = 1 if name in for_team else (0 if name in against_team else 2)
                    td = row.find('td', class_='standing-table__cell is-hidden--bp15 is-hidden--bp35')
                    form = td.find('div', class_='standing-table__form')
                    spans = form.find_all('span')
                    titles = [span['title'] for span in spans]
                    last_five_results = [result.strip() for result in titles[:5]]
                    last_five_record = [0, 0, 0]  # Wins, Draws, Losses
                    for result in last_five_results:
                        if name in result:
                            goals = result.split('-')
                            if len(goals) == 2:
                                if name in goals[0]:
                                    team_match = int(goals[0][-1].strip())
                                    opp_match = int(goals[1][0].strip())
                                else:
                                    team_match = int(goals[1][0].strip())
                                    opp_match = int(goals[0][-1].strip())
                                if team_match > opp_match:
                                    last_five_record[0] += 1  # Wins
                                elif team_match == opp_match:
                                    last_five_record[1] += 1  # Draws
                                else:
                                    last_five_record[2] += 1  # Losses
                    team_form = last_five_record[0] - last_five_record[2]
                    # Adding the numerical features to the data list
                    data.append([name, played, won, drawn, lost, gf, ga, gd, points,
                                 last_five_record[0], last_five_record[1], last_five_record[2], team_form,
                                 outcome])


    df = pd.DataFrame(data, columns=["Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points", 
                                     "Last_5_W", "Last_5_D", "Last_5_L","Team_Form", "Outcome"])
                                
    # -----------------------------------------------------------------------------------------------------
    df = df.query('Team_Form >= 3 or Team_Form <= -2')

    # Encoding the labels
    le = LabelEncoder()
    le.fit(df["Team"])

    # Spliting the data into training and testing sets
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Transforming the labels
    X_train["Team"] = le.transform(X_train["Team"])
    X_test["Team"] = le.transform(X_test["Team"])

    return X_train, X_test, y_train, y_test, le

#-------------------------------------------------------------------------------------------------------------------------

def train_model():
    try:
        X_train, X_test, y_train, y_test, le = models()
        # Initialize the model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Fitting the model to the training data
        rf_model.fit(X_train, y_train)

        # Evaluating the model on the testing data
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")

        return rf_model

    except Exception as e:
        print(f"Error: {e}")
        return None

# ----------------------------------------------------------------------

def prediction():
    try:
    # Train the model
        X_train, X_test, y_train, y_test, le = models()
        rf_model = train_model()

        # Getting feature importances
        importances = rf_model.feature_importances_
        features = X_train.columns

        # Finding the indices of the most important features
        indices = np.argsort(importances)[::-1]

        # Getting predictions for all the data
        all_data = pd.concat([X_train, X_test])
        all_predictions = rf_model.predict(all_data)
        all_outcome = pd.concat([y_train, y_test])

        # Getting the team names and outcomes
        team_names = le.inverse_transform(all_data["Team"])
        outcomes =  le.inverse_transform(all_outcome)
        predictions = le.inverse_transform(all_predictions)

        # Creating a dataframe with team names and their predicted outcomes
        team_predictions = pd.DataFrame({"Team": team_names, "Outcome": outcomes, "Prediction": predictions})
        
        # Printing the teams to be considered as favorites
        for_team = team_predictions[le.transform(team_predictions["Prediction"]) == 1]["Team"].values

        # Printing the teams with poor form
        against_team = team_predictions[le.transform(team_predictions["Prediction"]) == 0]["Team"].values

        # Printing the teams that can potentially win any match
        any_win = team_predictions[le.transform(team_predictions["Prediction"]) == 2]["Team"].values
    
        return for_team, against_team, any_win
        
    except Exception as e:
        print(f"Error: {e}")
        return None
