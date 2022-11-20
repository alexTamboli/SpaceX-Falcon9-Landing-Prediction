import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv("Data/dataset_part_2.csv")

def get_input(serialList, landingPads, launchSites, orbits):
    Fnum = int(input("Enter Flight Number (>170): "))
    mass = float(input("\nEnter Payload Mass (in kg): "))
    
    print("\nNOTE: Inputs are case-sensitive, make sure your input matches the options.")
    
    print("\nEnter orbit from below given list.")
    i = 0
    for ele in orbits:
        print("    " + str(i) + ". " + ele)
        i = i + 1
    orbit = input("=> ")
    
    print("\nEnter Launch Site from below given list.")
    i = 0
    for ele in launchSites:
        print("    " + str(i) + ". " + ele)
        i = i + 1
    launchSite = input("=> ")
    
    flights = int(input("\nEnter number of Flights (>0): "))
    gridfins = bool(input("\nEnter Grid Fins availibility (0 or 1): "))
    reused = bool(input("\nEnter if reused (0 or 1): "))
    legs = bool(input("\nDoes stage-1 have legs? (0 or 1): "))
    
    print("\nEnter Landing Pad from below given list.")
    i = 0
    for ele in landingPads:
        print("    " + str(i) + ". " + ele)
        i = i + 1
    landingPad = input("=> ")
    
    block = float(input("\nEnter number of Blocks (>0): "))
    reusedcount = int(input("\nIf Rocket is reused give its count (0 - if not used): "))
    
    counter = 0
    while True:
        serial = input("\nEnter Rocket Serial Number (ex B1003): ")
        counter = counter + 1
        if (serial in serialList) :
            break
        else:
             print("Not a valid Rocket Serial Number. Try Again.")
        if counter >= 10: 
            print("Too many wrong attempts. Please Rerun.\n For help go to esjksf.com")
            break
    
    res = [Fnum, mass, orbit, launchSite, flights, gridfins, reused, legs, landingPad, block, reusedcount, serial]  
    return res



# Selected features after Exploratory Data Analysis
param = ['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']
features = df[param] # Previous data
temp = pd.DataFrame(columns = param)

serialGrp = list(features.groupby('Serial').groups.keys())
launchGrp = list(features.groupby('LaunchSite').groups.keys())
landSiteGrp = list(features.groupby('LandingPad').groups.keys())
orbitGrp = list(features.groupby('Orbit').groups.keys())

# Getting input
inputs = get_input(serialList = serialGrp, landingPads = landSiteGrp, orbits = orbitGrp, launchSites = launchGrp)

dic = {'FlightNumber': inputs[0], 
       'PayloadMass': inputs[1], 
       'Orbit': inputs[2], 
       'LaunchSite': inputs[3], 
       'Flights': inputs[4], 
       'GridFins': inputs[5], 
       'Reused': inputs[6], 
       'Legs': inputs[7], 
       'LandingPad': inputs[8], 
       'Block': inputs[9], 
       'ReusedCount': inputs[10], 
       'Serial': inputs[11]}

# Append user input
features = features.append(dic,ignore_index=True)

features["FlightNumber"] = features["FlightNumber"].astype(int)
features["PayloadMass"] = features["PayloadMass"].astype(float)
features["Flights"] = features["Flights"].astype(int)
features["GridFins"] = features["GridFins"].astype(bool)
features["Reused"] = features["Reused"].astype(bool)
features["Legs"] = features["Legs"].astype(bool)
features["Block"] = features["Block"].astype(float)
features["ReusedCount"] = features["ReusedCount"].astype(int)

oh_orbit = pd.get_dummies(features["Orbit"])
oh_launch = pd.get_dummies(features["LaunchSite"])
oh_landing = pd.get_dummies(features["LandingPad"])
oh_serial = pd.get_dummies(features["Serial"])
remainder = features[["FlightNumber","PayloadMass", "Flights", "GridFins", "Reused", "Legs", "Block","ReusedCount"]]
features_one_hot = pd.concat([oh_launch, oh_landing, oh_serial, oh_orbit], axis=1)
features_one_hot.astype('float64')

features_one_hot.to_csv('Data/predict.csv', index=False)
print("(4) Data exported locally.")