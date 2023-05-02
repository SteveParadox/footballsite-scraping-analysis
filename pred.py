from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import numpy as np



from _ext import teams, train_model


# Getting the fixtures of the already predicted teams
def get_fixtures():
    start_date = datetime(2023, 4, 28)  
    end_date = datetime(2023, 4, 30)    
    delta = timedelta(days=1)

    match_fix = []
    compiled_for = set()
    compiled_against = set()
    compiled_any = set()
    while start_date <= end_date:
        date_str = start_date.strftime('%d-%B-%Y')
        url = f'https://www.skysports.com/football/fixtures-results/{date_str}'
        response = requests.get(url)

        soup = BeautifulSoup(response.content, 'html.parser')
        fixtures = soup.find_all('div', class_='fixres__item')
        for fixture in fixtures:
            league = fixture.find_previous_sibling('h5').text.strip()
            team1 = fixture.find('span', class_='swap-text__target').text.strip()
            team2 = fixture.find_all('span', class_='swap-text__target')[1].text.strip()
            time = fixture.find('span', class_='matches__date').text.strip()
            match_fix.append(f'{team1} vs {team2}')
        
        start_date += delta
        
        with open('for_selections.txt', 'r') as f:
            selection_set = set(line.strip() for line in f)
        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_for.add(item)
                
        with open('against_selections.txt', 'r') as f:
            selection_set = set(line.strip() for line in f)
        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_against.add(item)

        with open('any_selections.txt', 'r') as f:
            selection_set = set(line.strip() for line in f)

        for selection in selection_set:
            for item in match_fix:
                if selection in item:
                    compiled_any.add(item)
    
    return compiled_for, compiled_against, compiled_any


def arrange_fixture():

    # Removing unwanted data and organizing the fixtures into sections
    compiled_for, compiled_against, compiled_any = get_fixtures()
    compiled_for = {match for match in compiled_for if 'Ladies' not in match and 'Women' not in match}
    compiled_against = {match for match in compiled_against if 'Ladies' not in match and 'Women' not in match}
    compiled_any = {match for match in compiled_any if 'Ladies' not in match and 'Women' not in match}

    return compiled_for, compiled_against, compiled_any



def comparison():
    compiled_for, compiled_against, compiled_any = arrange_fixture() # Cleaned Fixtures
    for_team, against_team, any_win, data = teams() # league tables

    compiled_for = [[s.strip() for s in item.split('vs')] for item in compiled_for]
    compiled_against = [[s.strip() for s in item.split('vs')] for item in compiled_against]
    compiled_any = [[s.strip() for s in item.split('vs')] for item in compiled_any]

    comp_ = []

    for pair in compiled_against:
        sublist = []
        for item in pair:
            for row in data:
                if item in row:
                    sublist.extend(row)
        if len(sublist) == 26:
            comp_.append(sublist)
        else:
            print(f"Skipping pair {pair} because sublist has length {len(sublist)}")

    # Convert the list of stats to a pandas DataFrame
    df = pd.DataFrame(comp_, columns=["Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points", 
                                     "Last_5_W", "Last_5_D", "Last_5_L","Team_Form", "Away_Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points", 
                                     "Last_5_W", "Last_5_D", "Last_5_L","Team_Form"])
    
    #df = pd.get_dummies(df, columns=["Team"])
    
    # Encoding the labels

    le = LabelEncoder()
    unique_labels = set(df["Team"]) | set(df["Away_Team"])
    le.fit(list(unique_labels))

    
    y = df["Team_Form"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Encoding the features
    X_train["Team"] = X_train["Team"].apply(lambda x: le.transform([x])[0] if x in unique_labels else -1)
    X_train["Away_Team"] = X_train["Away_Team"].apply(lambda x: le.transform([x])[0] if x in unique_labels else -1)

    X_test["Team"] = X_test["Team"].apply(lambda x: le.transform([x])[0] if x in unique_labels else -1)
    X_test["Away_Team"] = X_test["Away_Team"].apply(lambda x: le.transform([x])[0] if x in unique_labels else -1)

    print(X_train)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

  

     # Convert each string value to a unique integer value
    le = LabelEncoder()
    for i in range(len(compiled_against)):
        for j in range(len(compiled_against[i])):
            compiled_against[i][j] = le.fit_transform([compiled_against[i][j]])[0]

    # Convert comp_ to a numpy array
    compiled_against = np.array(compiled_against)

    # Make predictions using the trained model
    predictions = rfc.predict(comp_)
    print(predictions)

    for i in range(len(compiled_against)):
        team1, team2 = le.inverse_transform(compiled_against[i][0]), le.inverse_transform(compiled_against[i][1])
        outcome = "Win" if np.any(predictions[i] == 2) else "Loss"
        print(f"{team1} vs {team2}: {outcome}")
        
    return 




print(comparison())

