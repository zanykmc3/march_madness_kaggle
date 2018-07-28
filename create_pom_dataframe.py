import pandas as pd
import xlrd
from string import digits
pom_dataframe = pd.read_excel("PomStats-Test-2018.xlsx")
teams_dataframe = pd.read_csv("DataFiles/Teams.csv")

pom_dataframe["TeamName"] = "here"
pom_dataframe["TeamID"] = 0
for index, row in pom_dataframe.iterrows():
    team = (row["Team"])
    result = team.split()
    name  = ""
    ctr = 0
    if(result[0] == "Saint"):
        result[0] = "St"
    
    if(len(result) > 2):
        for i  in range(0,len(result)-1):
            if(result[i] == "St."):
                result[i] = "St"
            if(ctr>0):
                name = name + " " + result[i]
            else:
                name = name + result[i]
            ctr = ctr+1
        if(result[i+1].isdigit()==False):
            if(result[i+1] == "St."):
                result[i+1] = "St"
            name = name + " " + result[i+1]
    else:
        if(len(result) == 2 and result[1].isdigit() == False):
            if(result[1] == "St."):
                result[1] = "St"
            name = result[0] + " " + result[1]
        else:
            name = result[0]
            
    pom_dataframe.set_value(index, 'TeamName', name)

for index, row in pom_dataframe.iterrows():
    team = (row["TeamName"])
    for index2, row_teams in teams_dataframe.iterrows():
        teamname = row_teams["TeamName"]
        if(team == (teamname)):
            pom_dataframe.set_value(index, 'TeamID', row_teams["TeamID"])

pom_dataframe.to_csv("Pom2018.csv",index=False)
