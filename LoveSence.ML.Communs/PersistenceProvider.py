import pymongo
import MongoDBConfig as config_db

class MongoDB:
  def __init__(self, store):
    environement = config_db.environement

    cfg = config_db.db.get(environement, "{}")
    user = cfg['user']
    pwd = cfg['pwd']
    db = cfg[store][0]
    col = cfg[store][1]

    self.client = pymongo.MongoClient(cfg.get('connection_string').format(user, pwd, db))
    self.database = self.client[db]
    self.collection = self.database[col]

  def ajouter_document(self, document):
    return self.collection.insert_one(document)

  def ajouter_documents(self, documents):
    return self.collection.insert_many(documents)

  def supprimer_document(self, document):
    return self.collection.delete_one(document)

  def supprimer_documents(self, documents):
    return self.collection.delete_many(documents)

  def filtre_collection(self, query, sort, limit):
    return self.collection.find(query).sort(sort).limit(limit)

  def obtenir_base_de_donnees():
     return  myclient.list_database_names()

  def obtenir_collections():
     return  myclient.list_collection_names()
