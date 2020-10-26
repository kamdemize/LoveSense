
from time import time
from datetime import datetime
import pickle 

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider

def obtenir_experiences(request):
    try :
        ml_experiences = db_provider.MongoDB('experiences').filtre_collection({}, [('date_experience', -1)], 20)
        if ml_experiences.count()==0:
            return None
        
        docs = list(ml_experiences)
        return [{"date_experience": doc['date_experience'], "type": doc['type'], "code_experience": doc['code_experience'], "score": doc['score'], "time_training": doc['time_training'] , "time_test": doc['time_test'], "erreur_experience": doc['erreur_experience']} for doc in docs]

    except Exception as e:
        return {'status': "erreur", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}