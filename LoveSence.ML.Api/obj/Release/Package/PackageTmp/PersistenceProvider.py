import pymongo


MONGO_CONNEXION_STRING = "mongodb+srv://<username>:<password>@cluster0.rvdzr.mongodb.net/<dbname>?retryWrites=true&w=majority"

class MongoDB:
  def __init__(self, database, collection):
    connection_string = "mongodb+srv://{}:{}@cluster0.rvdzr.mongodb.net/{}?retryWrites=true&w=majority".format('superuser','pDnp94IO4CNZv7zy', database)
    self.client = pymongo.MongoClient(connection_string)
    
    if database in self.client.list_database_names():
        self.database = self.client[database]
    else:
        self.database = self.client[database]

    if collection in self.database.list_collection_names():
        self.collection = self.database[collection]
    else:
        self.collection = self.database[collection]
 
  def ajouter_document(self, document):
    return self.collection.insert_one(document)

  def filtre_collection(self, query, sort, limit):
    return self.collection.find(query).sort(sort).limit(limit)

  def obtenir_base_de_donnees():
     return  myclient.list_database_names()

  def obtenir_collections():
     return  myclient.list_collection_names()
