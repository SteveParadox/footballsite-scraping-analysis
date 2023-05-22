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
import time

from urldata import urls

start = time.time()


def teams():
    for_team = []
    against_team = []
    any_win = []
    data = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")
        table = soup.find("table")
        rows = table.find_all("tr")
        for row in rows[1:]:
            columns = row.find_all("td")
            name = row["data-team-name"]
            played = int(row.find("td", {"class": "widget-match-standings__matches-played"}).text.strip())
            won = int(row.find("td", {"class": "widget-match-standings__matches-won"}).text.strip())
            drawn = int(row.find("td", {"class": "widget-match-standings__matches-drawn"}).text.strip())
            lost = int(row.find("td", {"class": "widget-match-standings__matches-lost"}).text.strip())
            gf = int(row.find("td", {"class": "widget-match-standings__goals-for"}).text.strip())
            ga = int(row.find("td", {"class": "widget-match-standings__goals-against"}).text.strip())
            gd = int(row.find("td", {"class": "widget-match-standings__goals-diff"}).text.strip())
            points = int(row.find("td", {"class": "widget-match-standings__pts"}).text.strip())
            dd =row.find("td", {"class": "widget-match-standings__last-five"}).text.strip()
            _data = dd.strip().split('\n')
            row_list = []

            for row in _data:
                row_list = row.split()
            # Converting the last five results into a win-draw-loss record
            last_five_record = [0, 0, 0]  # Wins, Draws, Losses
            last_five_record[0] = row_list.count("W")  # Wins
            last_five_record[1] = row_list.count("D") # Draws
            last_five_record[2] = row_list.count("L") # Loss

            # Checking if the team has less than 6 losses or more than 17 losses
            team_form = last_five_record[0] - last_five_record[2]
            if played > 20:
                data.append([name, played, won, drawn, lost, gf, ga, gd, points,
                            last_five_record[0], last_five_record[1], last_five_record[2], team_form,
                            ])

                if lost <= 9 or won >= 18:
                    for_team.append(name)
                elif lost >= 14 or won <= 7:
                    against_team.append(name)
                elif drawn < 12:
                    any_win.append(name)
        else:
            pass
    return for_team, against_team, any_win, data

# --------------------------------------------------------------------------------------------------------

def models():
    for_team, against_team, any_win, data = teams()
    data_ = []  
    # Add the data for each team to the list
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")
        table = soup.find("table")
        rows = table.find_all("tr")
        for row in rows[1:]:
            columns = row.find_all("td")
            name = row["data-team-name"]
            if name in for_team or name in against_team or name in any_win :
                played = int(row.find("td", {"class": "widget-match-standings__matches-played"}).text.strip())
                won = int(row.find("td", {"class": "widget-match-standings__matches-won"}).text.strip())
                drawn = int(row.find("td", {"class": "widget-match-standings__matches-drawn"}).text.strip())
                lost = int(row.find("td", {"class": "widget-match-standings__matches-lost"}).text.strip())
                gf = int(row.find("td", {"class": "widget-match-standings__goals-for"}).text.strip())
                ga = int(row.find("td", {"class": "widget-match-standings__goals-against"}).text.strip())
                gd = int(row.find("td", {"class": "widget-match-standings__goals-diff"}).text.strip())
                points = int(row.find("td", {"class": "widget-match-standings__pts"}).text.strip())
                dd =row.find("td", {"class": "widget-match-standings__last-five"}).text.strip()
                _data = dd.strip().split('\n')

                row_list = []

                for row in _data:
                    row_list = row.split()
                outcome = 1 if name in for_team else (0 if name in against_team else 2)
                last_five_record = [0, 0, 0]  # Wins, Draws, Losses
                last_five_record[0] = row_list.count("W")  # Wins
                last_five_record[1] = row_list.count("D") # Draws
                last_five_record[2] = row_list.count("L") # Loss

                # Checking if the team has less than 6 losses or more than 17 losses
                team_form = last_five_record[0] - last_five_record[2]
                
                # Adding the numerical features to the data list
                data_.append([name, played, won, drawn, lost, gf, ga, gd, points,
                                last_five_record[0], last_five_record[1], last_five_record[2], team_form,
                                outcome])

    
    df = pd.DataFrame(data_, columns=["Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points", 
                                     "Last_5_W", "Last_5_D", "Last_5_L","Team_Form", "Outcome"])
                                
    # -----------------------------------------------------------------------------------------------------
    df = df.query('Team_Form > 2 or Team_Form < -1')

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

    return X_train, X_test, y_train, y_test, le, data_, data

#-------------------------------------------------------------------------------------------------------------------------

def train_model():
    try:
        X_train, X_test, y_train, y_test, le, data_, data = models()
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

        return rf_model, X_train, X_test, y_train, y_test, le, data_, data

    except Exception as e:
        print(f"Error: {e}")
        return None

# ----------------------------------------------------------------------

def prediction():
    try:
    # Train the model
        rf_model, X_train, X_test, y_train, y_test, le, data_, data = train_model()

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
    
        return for_team, against_team, any_win, data_, data
        
    except Exception as e:
        print(f"Error: {e}")
        return None
