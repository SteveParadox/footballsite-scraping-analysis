from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import pandas as pd

from _ext import teams, prediction


# Getting the fixtures of the already predicted teams
def get_fixtures():
    try:
        for_team, against_team, any_win = prediction()
        start_date = datetime.today() + timedelta(days=2)
        end_date = start_date + timedelta(days=7)
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
                team_1 = fixture.find('span', class_='swap-text__target').text.strip()
                team_2 = fixture.find_all('span', class_='swap-text__target')[1].text.strip()
                time = fixture.find('span', class_='matches__date').text.strip()
                match_fix.append(f'{team_1} vs {team_2}')
            
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
        
        return compiled_for, compiled_against, compiled_any
    except Exception as e:
        print(f"Error: {e}")
        return None 


def arrange_fixture():
    # Removing unwanted data and organizing the fixtures into sections
    compiled_for, compiled_against, compiled_any = get_fixtures()
    compiled_for = {match for match in compiled_for if 'Ladies' not in match and 'Women' not in match}
    compiled_against = {match for match in compiled_against if 'Ladies' not in match and 'Women' not in match}
    compiled_any = {match for match in compiled_any if 'Ladies' not in match and 'Women' not in match}

    return compiled_for, compiled_against, compiled_any

def my_pick():
    compiled_for, compiled_against, compiled_any = arrange_fixture() # Cleaned Fixtures
    for_team, against_team, any_win, data = teams() # league tables
    
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
    
    for i, j in zip(comp_, compiled_against):
        if i[0] in j and i[-1] >= 3 and i[15] >= 14:
            pick_against.append(i[0])
        elif i[13] in j and i[12] >= 3 and i[2] >= 14: 
            pick_against.append(i[13])
    for i, j in zip(_comp, compiled_for):
        if i[0] in j and i[-1] <= -2 and i[15] <= 13:
            pick_for.append(i[0])
        elif i[13] in j and i[12] <= -2 and i[2] <= 13:
            pick_for.append(i[13])
    
    return pick_for, pick_against



if __name__ == "__main__":
    pick_for, pick_against = my_pick()
    print(pick_for)
    print(pick_against)

        