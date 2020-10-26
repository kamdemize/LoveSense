
from sklearn.model_selection import train_test_split

#UserModule
import sys
import os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + '\\LoveSence.ML.Communs')
import AccessCorpus as corpus
import CleanningCorpus as cleanning



def obtenir_donnees_entrainement():
    echantillon_donnees = echantilloner_donnees(cleanning.clean(corpus.obtenir()))
    return (echantillon_donnees[0], echantillon_donnees[2])

def obtenir_donnees_entrainement_test():
    echantillon_donnees = echantilloner_donnees(cleanning.clean(corpus.obtenir()))
    return echantillon_donnees

def echantilloner_donnees():
    doonnees_corpus = cleanning.clean(corpus.obtenir())
    # Split the data into train & test sets:
    X = doonnees_corpus['clean_message']
    y = doonnees_corpus['alabel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 42)
    return (X_train, X_test, y_train, y_test)
