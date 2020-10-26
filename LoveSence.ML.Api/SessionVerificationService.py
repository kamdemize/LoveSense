
from time import time
from datetime import datetime
import pickle 

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider

def obtenir_sessions(request):
    try :
        sessions = db_provider.MongoDB('sessions').filtre_collection({}, [('date_session', -1)], 20)
        if sessions.count()==0:
            return None
        
        docs = list(sessions)
        return [{"date_session": doc['date_session'], "verdict": doc['verdict'], "text": doc['message']['text'], "score": -1} for doc in docs]

    except Exception as e:
        return {'status': "erreur", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}