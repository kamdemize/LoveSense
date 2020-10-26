from time import time
from datetime import datetime
import pickle 

#Analytic
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

##ML
##ML.Vectorize
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
##ML.Selection
#from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
##ML.Reduction
#from sklearn.decomposition import TruncatedSVD 
#from sklearn.decomposition import NMF
#from sklearn.decomposition import KernelPCA
##ML.Models
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
##ML.Evaluate
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix,classification_report
#from sklearn import metrics
##Optimisation
#from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.model_selection import GridSearchCV

#UserModule
#import PersistenceProvider as db_provider

def gerer_verification(request, message):
    return {'status': "sucess", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}
    #try :
    #    if (message and message['text'] and not message['text'].isspace()):
    #        experience = obtenir_experience_optimal()

    #        if experience:
    #            model = pickle.loads(experience['model_pickle'])
    #            verdict = message_legitime(model, message)
    #            logger_session_verication(request, message, verdict, str(experience['_id']))

    #            return {'status': "succes", 'verdict': verdict, 'score' : experience['score']}
    #        else :
    #            return {'status': "error", 
    #                    'msg': "Le système est en cours de maintenance, essayer plutard"}
    #    else:
    #        return {'status': "error", 
    #                'msg': "Le message n'est pas valide (ex: {'test' : 'le message à véfifier ...'})"}

    #except Exception as e:
    #    return {'etat': "erreur", 
    #                    'msg': "Le système est en cours de maintenance, essayer plutard"}

#def obtenir_experience_optimal():
#    experiences = db_provider.MongoDB('db_dev','ml_dev').filtre_collection({}, [('score', -1)], 1)
#    if experiences.count()==0:
#       return None
        
#    return experiences[0]

#def message_legitime(model, message):
#    verdict = model.predict(message)
#    return verdict[0] == 'legitime'

#def logger_session_verication(request, message, verdict, experience_id):    
#    session = {
#            'request_pickle' : pickle.dumps(request),
#            'message' : message,
#            'verdict' : verdict,
#            'experience_id' : experience_id,
#            'date_session' : datetime.now()
#        }
#    mongo_db = db_provider.MongoDB('db_dev','ml_sessions_dev').ajouter_document(session)
