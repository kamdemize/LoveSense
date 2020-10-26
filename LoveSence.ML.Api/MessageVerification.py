from time import time
from datetime import datetime
import pickle 

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider

def gerer_verification(request, message):
    try :
        if (message and message['text'] and not message['text'].isspace()):
            experience = obtenir_experience_optimal()

            if experience:
                model = pickle.loads(experience['model_pickle'])
                verdict = message_legitime(model, message)
                logger_session_verication(request, message, verdict, str(experience['_id']))

                return {'status': "succes", 'verdict': verdict, 'score' : experience['score']}
            else :
                return {'status': "error", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}
        else:
            return {'status': "error", 
                    'msg': "Le message n'est pas valide (ex: {'test' : 'le message à véfifier ...'})"}

    except Exception as e:
        return {'status': "erreur", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}

def obtenir_experience_optimal():
    experiences = db_provider.MongoDB('experiences').filtre_collection({}, [('score', -1)], 1)
    if experiences.count()==0:
       return None
        
    return experiences[0]

def message_legitime(model, message):
    verdict = model.predict(message)
    return verdict[0] == 'legitime'

def logger_session_verication(request, message, verdict, experience_id):    
    session = {
            'request_pickle' : pickle.dumps(request),
            'message' : message,
            'verdict' : verdict,
            'experience_id' : experience_id,
            'date_session' : datetime.now()
        }
    mongo_db = db_provider.MongoDB('sessions').ajouter_document(session)
