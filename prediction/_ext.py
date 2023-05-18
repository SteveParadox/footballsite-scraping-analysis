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

start = time.time()
urls = [
    "https://www.goal.com/en-us/premier-league/table/6wubmo7di3kdpflluf6s8c7vs",
    "https://www.goal.com/en-us/ligue-1/table/57nu0wygurzkp6fuy5hhrtaa2",
    "https://www.goal.com/en-us/liga-profesional-argentina/table/581t4mywybx21wcpmpykhyzr3",
    "https://www.goal.com/en-us/a-league-men/table/xwnjb1az11zffwty3m6vn8y6",
    "https://www.goal.com/en-us/first-division-a/table/4zwgbb66rif2spcoeeol2motx",
    "https://www.goal.com/en-us/premijer-liga/table/4yngyfinzd6bb1k7anqtqs0wt",
    "https://www.goal.com/en-us/serie-a/table/scf9p4y91yjvqvg5jndxzhxj",
    "https://www.goal.com/en-us/primera-divisi%C3%B3n/table/2y8bntiif3a9y6gtmauv30gt",
    "https://www.goal.com/en-us/primera-b/table/bly7ema5au6j40i0grhl0pnub",
    "https://www.goal.com/en-us/csl/table/82jkgccg7phfjpd0mltdl3pat",
    "https://www.goal.com/en-us/primera-a/table/2ty8ihceabty8yddmu31iuuej",
    "https://www.goal.com/en-us/primera-b/table/by5nibd18nkt40t0j8a0j5yzx",
    "https://www.goal.com/en-us/hnl/table/1b70m6qtxrp75b4vtk8hxh8c3",
    "https://www.goal.com/en-us/czech-liga/table/bu1l7ckihyr0errxw61p0m05",
    "https://www.goal.com/en-us/superliga/table/29actv1ohj8r10kd9hu0jnb0n",
    "https://www.goal.com/en-us/liga-pro/table/6lwpjhktjhl9g7x2w7njmzva6",
    "https://www.goal.com/en-us/premier-league/table/8k1xcsyvxapl4jlsluh3eomre",
    "https://www.goal.com/en-us/championship/table/7ntvbsyq31jnzoqoa8850b9b8",
    "https://www.goal.com/en-us/league-one/table/3frp1zxrqulrlrnk503n6l4l",
    "https://www.goal.com/en-us/premier-league/table/2kwbbcootiqqgmrzs6o5inle5",
    "https://www.goal.com/en-us/league-two/table/bgen5kjer2ytfp7lo9949t72g",
    "https://www.goal.com/en-us/premier-league-2-division-one/table/75434tz9rc14xkkvudex742ui",
    "https://www.goal.com/en-us/premier-league-2-division-two/table/a0zpsx4vvgvn2kpxzg1bcciui",
    "https://www.goal.com/en-us/ligue-1/table/dm5ka0os1e3dxcp3vh05kmp33",
    "https://www.goal.com/en-us/ligue-2/table/4w7x0s5gfs5abasphlha5de8k",
    "https://www.goal.com/en-us/bundesliga/table/6by3h89i2eykc341oz7lv1ddd",
    "https://www.goal.com/en-us/2-bundesliga/table/722fdbecxzcq9788l6jqclzlw",
    "https://www.goal.com/en-us/premier-league/table/4jg7he1n3rb5dniq6hf49xorq",
    "https://www.goal.com/en-us/super-league-1/table/c0r21rtokgnbtc0o2rldjmkxu",
    "https://www.goal.com/en-us/premier-league/table/4rls982p5uzil6x30mhyhv9f3",
    "https://www.goal.com/en-us/nb-i/table/47s2kt0e8m444ftqvsrqa3bvq",
    "https://www.goal.com/en-us/nb-ii/table/beqqnubkv05mamuwvimeum015",
    "https://www.goal.com/en-us/indian-super-league/table/3oa9e03e7w9nr8kqwqc3tlqz9",
    "https://www.goal.com/en-us/i-league/table/4pohvulrkgzx38eoqse6b5cdg",
    "https://www.goal.com/en-us/isc-a/table/253foz8zjbecgiyhz4cgytxih",
    "https://www.goal.com/en-us/liga-1/table/117yqo02rs8dykkxpm274w3bd",
    "https://www.goal.com/en-us/serie-a/table/1r097lpxe0xn03ihb7wi98kao",
    "https://www.goal.com/en-us/fkf-premier-league/table/7wssxdqi4xihseeam8grqa2b8",
    "https://www.goal.com/en-us/j2-league/table/5z8v4mj6cjs9ex6hdrpourjzh",
    "https://www.goal.com/en-us/serie-b/table/8ey0ww2zsosdmwr8ehsorh6t7",
    "https://www.goal.com/en-us/serie-c/table/1zp1du9n4rj36p1ss9zbxtqfb",
    "https://www.goal.com/en-us/j1-league/table/8o5tv5viv4hy1qg9jp94k7ayb",
    "https://www.goal.com/en-us/super-league/table/eg6s9f1jj7jr6stmbosn0g6c8",
    "https://www.goal.com/en-us/liga-mx/table/2hsidwomhjsaaytdy9u5niyi4",
    "https://www.goal.com/en-us/botola-pro/table/1eruend45vd20g9hbrpiggs5u",
    "https://www.goal.com/en-us/eerste-divisie/table/1gwajyt0pk2jm5fx5mu36v114",
    "https://www.goal.com/en-us/eredivisie/table/akmkihra9ruad09ljapsm84b3",
    "https://www.goal.com/en-us/eliteserien/table/9ynnnx1qmkizq1o3qr3v0nsuk",
    "https://www.goal.com/en-us/division-profesional/table/5y0z0l2epprzbscvzsgldw8vu",
    "https://www.goal.com/en-us/primera-divisi%C3%B3n/table/a9vrdkelbgif0gtu3wxsr75xo",
    "https://www.goal.com/en-us/ekstraklasa/table/7hl0svs2hg225i2zud0g3xzp2",
    "https://www.goal.com/en-us/primeira-liga/table/8yi6ejjd1zudcqtbn07haahg6",
    "https://www.goal.com/en-us/stars-league/table/xaouuwuk8qyhv1libkeexwjh",
    "https://www.goal.com/en-us/liga-i/table/89ovpy1rarewwzqvi30bfdr8b",
    "https://www.goal.com/en-us/saudi-league/table/ea0h6cf3bhl698hkxhpulh2zz",
    "https://www.goal.com/en-us/championship/table/8t2o4huu2e48ij23dxnl9w5qx",
    "https://www.goal.com/en-us/league-one/table/6sxm2iln2w45ux498pty9miw8",
    "https://www.goal.com/en-us/league-two/table/6321dlqv4ziuwqte4xpohijtw",
    "https://www.goal.com/en-us/premiership/table/e21cf135btr8t3upw0vl6n6x0",
    "https://www.goal.com/en-us/psl/table/yv73ms6v1995b5wny16jcfi3",
    "https://www.goal.com/en-us/primera-divisi%C3%B3n/table/34pl8szyvrbwcmfkuocjm3r6t",
    "https://www.goal.com/en-us/segunda-divisi%C3%B3n/table/3is4bkgf3loxv9qfg3hm8zfqb",
    "https://www.goal.com/en-us/sudani-premier-league/table/2c01jrik7ggtta321pstz8tm4",
    "https://www.goal.com/en-us/allsvenskan/table/b60nisd3qn427jm0hrg9kvmab",
    "https://www.goal.com/en-us/super-league/table/e0lck99w8meo9qoalfrxgo33o",
    "https://www.goal.com/en-us/ligi-kuu-bara/table/9z5643nd06afqu01ea2wt8y4g",
    "https://www.goal.com/en-us/thai-league-1/table/iu1vi94p4p28oozl1h9bvplr",
    "https://www.goal.com/en-us/thai-league-2/table/bt24epydr1s8zc2x5xb0n9noc",
    "https://www.goal.com/en-us/1-lig/table/2o9svokc5s7diish3ycrzk7jm",
    "https://www.goal.com/en-us/s%C3%BCper-lig/table/482ofyysbdbeoxauk19yg7tdt",
    "https://www.goal.com/en-us/pro-league/table/f39uq10c8xhg5e6rwwcf6lhgc",
    "https://www.goal.com/en-us/mls/table/287tckirbfj9nb8ar2k9r60vn",
    "https://www.goal.com/en-us/vleague-1/table/aho73e5udydy96iun3tkzdzsi",
    "https://www.goal.com/en-us/premier-soccer-league/table/4azsryi40zahspm5h6d0f0pgl"
    
]

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
