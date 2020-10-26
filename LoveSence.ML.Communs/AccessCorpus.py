import os 
import pandas as pd

def obtenir_file():
    return os.path.dirname(__file__) + "\\corpus\\corpus.csv"    

def obtenir():
    return pd.read_csv(obtenir_file())

