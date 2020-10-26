import sys
import os
from pathlib import Path
from time import time
from datetime import datetime
import pickle 

#Analytic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ML
#ML.Vectorize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#ML.Selection
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
#ML.Reduction
from sklearn.decomposition import TruncatedSVD 
from sklearn.decomposition import NMF
from sklearn.decomposition import KernelPCA
#ML.Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#ML.Evaluate
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics
#Optimisation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#UserModule
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + '\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider
import DatasSampling as sampling

experience1_results = open("experience1_results_kn.txt","a") 
experience1_results.writelines('code_experience;type;score;tn;fp;fn;tp;fpr;fnr;precision;rappel;specificite;erreur_model;k;n_components;kernel;n_estimators;n_neighbors;t_training;t_test\n')

def definir_experiences():
    #Vectorisation
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True)
    tf = CountVectorizer(stop_words='english')
    ngram3 = CountVectorizer(analyzer='char', ngram_range=(3, 3))

    #Sélection des caracteristiques
    ki2 = SelectKBest(chi2)
    mic = SelectKBest(mutual_info_classif) 

    #Réduction des caracteristiques
    nmf = NMF() # Factorisation par matrices non-négatives
    lsa = TruncatedSVD()# latent semantic analysis
    ipca = KernelPCA() # Analyse en composantes principales

    #modèle standard
    lg = LogisticRegression() # logistic regression
    svm = SVC()  # svm
    rf = RandomForestClassifier() # random forest
    kn = KNeighborsClassifier() # KNeighbors

    #Expériences
    return {'vectoriser': [('tf',tf), ('tfidf',tfidf), ('ngram3',ngram3)],
            'selectinner': [('mic',mic), ('ki2',ki2)],
            'reduire': [('nmf',nmf), ('lsa',lsa), ('ipca',ipca)],
            'modeliser': [('kn',kn)]}
    #return {'vectoriser': [('tfidf',tfidf)],
    #        'selectinner': [('ki2',ki2)],
    #        'reduire': [('ipca',ipca)],
    #        'modeliser': [('lg',lg)]}
def conduire_experimentation(donnees_experiences, experiences):  
    X_train, X_test = donnees_experiences[0], donnees_experiences[1]
    y_train, y_test = donnees_experiences[2], donnees_experiences[3]

    vectoriser_methodes = experiences['vectoriser']
    selectinner_methodes = experiences['selectinner']
    reduire_methodes = experiences['reduire']
    modeliser_methodes = experiences['modeliser']

    resultats = []

    for v in vectoriser_methodes:
        for s in selectinner_methodes:
            for r in reduire_methodes:
                for m in modeliser_methodes:
                    resultats.append(conduire_experience(X_train, y_train, X_test, y_test, v, s, r, m))

    experience1_results.close() 
    return resultats

def conduire_experience(X_train, y_train, X_test, y_test, vectoriser, selectionner, reduire,  modeliser):
    n_components, k, t_training, t_test, n_estimators, n_neighbors = 0, 0, 0, 0,0,0
    score, tn, fn, tp, fp, erreur_model,fpr,fnr,precision,rappel,specificite = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    experience_impossible = False
    erreur_experience, kernel = '',''
    matrice_confusion = []
    ml_experience, autres_metriques = {}, {}
    methods_pickle, model_pickle = None, None
    code_experience = vectoriser[0] + '_' + selectionner[0] + '_' + reduire[0] + '_' + modeliser[0]
    print(str(datetime.now()) + ' Expérience:{0} - {1} {2} {3} {4}'.format(code_experience, vectoriser[0], selectionner[0], reduire[0], modeliser[0]))

    try:
        t0 = time()
        model = construire_modele(X_train, y_train, vectoriser, selectionner, reduire, modeliser)
        t_training = round(time() - t0, 3)
    except Exception as e:
        erreur_experience = str(e)
        experience_impossible = True
        pass
                    
    if (experience_impossible == False):
        t0 = time()
        metriques_evalusation = evaluer_experience(model,  X_test, y_test)
        t_test = round(time() - t0, 3)
        
        autres_metriques = metriques_evalusation['classification_report']
        tn = metriques_evalusation['tn']
        fn = metriques_evalusation['fn']
        tp = metriques_evalusation['tp']
        fp = metriques_evalusation['fp']
        erreur_model = metriques_evalusation['erreur_model']
        fpr = metriques_evalusation['fpr']
        fnr = metriques_evalusation['fnr']
        precision = metriques_evalusation['precision']
        rappel = metriques_evalusation['rappel']
        specificite = metriques_evalusation['specificite']
        classification_report = metriques_evalusation['classification_report']
        score = metriques_evalusation['taux_success']

        if 'sel__k' in model.best_params_.keys(): 
            k = model.best_params_['sel__k']
        if 'red__n_components' in model.best_params_.keys(): 
            n_components = model.best_params_['red__n_components']
        if 'mod__n_estimators' in model.best_params_.keys(): 
            n_estimators = model.best_params_['mod__n_estimators']
        if 'mod__kernel' in model.best_params_.keys(): 
            kernel = model.best_params_['mod__kernel']
        if 'mod__n_neighbors' in model.best_params_.keys(): 
            n_neighbors = model.best_params_['mod__n_neighbors']  

        methods_pickle = pickle.dumps({
                'vectorizer' : vectoriser[1],
                'selector' : selectionner[1],
                'reductor' : reduire[1],
                'classifier' : modeliser[1]
            })  
        model_pickle = pickle.dumps(model)

        ml_experience = {'date_experience': datetime.now(), 'type': 'I', 'methods_pickle': methods_pickle, 'code_experience' :code_experience,
           'score':score , 'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp, 'erreur_model': erreur_model, 'fpr': fpr, 'fnr': fnr  , 'precision': precision  , 'rappel': rappel  , 
           'specificite': specificite, 'autres_metriques' :  autres_metriques,
           'n_components': n_components, 'k': k, 'n_estimators': n_estimators, 'kernel': kernel, 'n_neighbors': n_neighbors,
           'erreur_experience': erreur_experience, 'time_training': t_training , 'time_test' : t_test, 'model_pickle' : model_pickle}

        experience1_results.writelines(str(code_experience) + ';' + 'I' + ';' + str(score) + ';' + str(tn) + ';' + str(fp) + ';' + str(fn) + ';' + str(tp) + ';' + str(fpr) + ';' + str(fnr) + ';' + str(precision) + ';' + str(rappel) + ';' + str(specificite) + ';' + str(erreur_model) + ';' + str(k) + ';' + str(n_components)  + ';' +  str(kernel) + ';' + str(n_estimators) + ';' + str(n_neighbors) + ';' + str(t_training) + ';' + str(t_test) + '\n')
        #mongo_db = db_provider.MongoDB('experiences').ajouter_document(ml_experience)

    return ml_experience

def construire_modele(X_train, y_train, vectoriser, selectionner, reduire, modeliser):
    pipe = Pipeline([('vec', vectoriser[1]),
                     ('sel', selectionner[1]),
                     ('red', reduire[1]),
                     ('mod', modeliser[1])])
    partition_cross_validation, parrallele_job = 3, 1
    hyper_parameters_selector_and_reductor = {
    'sel__k': [2000, 2500],
    'red__n_components': [200, 250] }

    hyper_parameters = {**hyper_parameters_selector_and_reductor, 
                        **obtenir_hyper_parameters_classifier(modeliser[0])}
    search = GridSearchCV(pipe, hyper_parameters, cv=partition_cross_validation, n_jobs=parrallele_job).fit(X_train, y_train)

    return search 

def evaluer_experience(model, X_test, y_test):
    y_predic = model.predict(X_test)
    matrice_confusion = confusion_matrix(y_test, y_predic)
    true_negative = matrice_confusion[0,0]
    false_negative = matrice_confusion[1,0]
    true_positive = matrice_confusion[1,1]
    false_positive = matrice_confusion[0,1]

    resultats_evaluation = {
        'tn' : true_negative,
        'fn' : false_negative,
        'tp' : true_positive,
        'fp' : false_positive,
        'erreur_model' : false_positive + false_negative,
        'fpr': round((false_positive / (false_positive + true_negative)),3),
        'fnr': round((false_negative / (false_negative + true_positive)),3),
        'precision': round((true_positive / (true_positive + false_positive)),3),
        'rappel': round((true_positive / (true_positive + false_negative)),3),
        'specificite': round((true_negative / (true_negative + false_positive)),3),
        'classification_report' : classification_report(y_test, y_predic, output_dict =True),
        'taux_success' : round(metrics.accuracy_score(y_test, y_predic),3)
       }

    return resultats_evaluation

def resultats_experience_to_panda(array):
    resultats_panda = pd.DataFrame(array, columns = ['date_experience', 'type', 'methods_pickle', 'code_experience' ,'score','matrice_confusion_pickle','autres_metriques','n_components','k','erreur_experience','time_training', 'time_test','model_pickle'])
    return resultats_panda

def editier_experimentation(resultats):
    result = resultats[resultats.score != -1]
    result = result.sort_values(by=['score'])
    scores = result['score'].values.tolist()
    experiences = result['code_experience'].values.tolist()
    fig, ax = plt.subplots()
    index = np.arange(len(experiences))
    plt.barh(index, scores, color= 'blue', align='center')
    ax.set_yticks(index)
    ax.set_yticklabels(experiences)
    ax.invert_yaxis()
    ax.set_xlabel('Score')
    ax.set_title("Score selon l'expérience")
    plt.show()

def obtenir_hyper_parameters_classifier(classifier):
    hyper_parameters = {}

    if classifier == 'rf':
        hyper_parameters = { 'mod__n_estimators': [500, 750, 1000]}

    if classifier == 'svm':
        hyper_parameters = { 'mod__kernel': ['linear', 'poly', 'rbf','sigmoid']}

    if classifier == 'kn':
        hyper_parameters = {'mod__n_neighbors': [3, 5]}

    return hyper_parameters

bilan_experimentation = conduire_experimentation(sampling.echantilloner_donnees(), definir_experiences())
a = 1
#mongo_db = db_provider.MongoDB('experiences').ajouter_documents(bilan_experimentation)
#editier_experimentation(resultats_experience_to_panda(bilan_experimentation))