
from time import time
from datetime import datetime
import pickle 

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider

def obtenir_corpus(request):
    try :
        corpus = db_provider.MongoDB('corpus').filtre_collection({}, [('date_creation', -1)], 20)
        if corpus.count()==0:
            return None
        
        docs = list(corpus)
        return [{"date_creation": doc['date_creation'], "label": doc['label'], "text": doc['text']} for doc in docs]


    except Exception as e:
        return {'status': "erreur", 
                        'msg': "Le système est en cours de maintenance, essayer plutard"}

#corpus = cleanning.clean(corpus.obtenir()).values.tolist()
#jsons = []
#for doc in corpus:
#    json = {'date_creation' : datetime.now(), 
#            'label' : doc[0], 
#            'text' : doc[1] 
#            }
#    jsons.append(json)

#mongo_db = db_provider.MongoDB('corpus').ajouter_documents(jsons)