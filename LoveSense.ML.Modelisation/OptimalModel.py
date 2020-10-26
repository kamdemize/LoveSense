import sys
import os
from pathlib import Path
from time import time
from datetime import datetime
import json  
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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
#UserModule
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + '\\LoveSence.ML.Communs')
import PersistenceProvider as db_provider
import DatasSampling as sampling

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
    return {'vectoriser': [('tf', tf), ('tfidf', tfidf), ('ngram3', ngram3)],
            'selectinner': [('mic', mic), ('ki2', ki2)],
            'reduire': [('nmf', nmf), ('lsa', lsa), ('ipca', ipca)],
            'modeliser': [('rf', rf)]}

    #return {'vectoriser': [('tfidf',tfidf)],
    #        'selectinner': [('mic',mic)],
    #        'reduire': [('nmf',nmf)],
    #        'modeliser': [('rf',rf)]}

def conduire_experience(donnees_experiences, experiences):
    X_train, X_test = donnees_experiences[0], donnees_experiences[1]
    y_train, y_test = donnees_experiences[2], donnees_experiences[3]

    vectorizers = experiences['vectoriser']
    selectors = experiences['selectinner']
    reductors = experiences['reduire']
    classifiers = experiences['modeliser']
    
    n_components, k, t_training, t_test, n_estimators, n_neighbors =  0, 0, 0, 0, 0, 0
    score, tn, fn, tp, fp, erreur_model,fpr,fnr,precision,rappel,specificite = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    experience_impossible = False
    erreur_experience, kernel = '', ''
    matrice_confusion = []
    autres_metriques = {}
    methods_pickle, model_pickle = None, None
    steps = {}
    
    for classifier in classifiers:
        experience1_results = open("experience1_results_optimal_" + classifier[0] + ".txt","a")  
        experience1_results.writelines('code_experience;type;score;tn;fp;fn;tp;fpr;fnr;precision;rappel;specificite;erreur_model;k;n_components;kernel;n_estimators;n_neighbors;t_training;t_test\n')

        try:
            code_experience = 'fusion' + '_' + classifier[0]
            print(str(datetime.now()) + code_experience)
            
            t0 = time()
            model = construire_modele(X_train, y_train,  X_test, y_test, vectorizers, selectors, reductors, classifier)
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

            if 'selector__ki2__k' in model.best_params_.keys(): 
                k = model.best_params_['selector__ki2__k']
            if 'selector__mic__k' in model.best_params_.keys(): 
                k = model.best_params_['selector__mic__k']

            if 'reductor__nmf__n_components' in model.best_params_.keys(): 
                n_components = model.best_params_['reductor__nmf__n_components']
            if 'reductor__lsa__n_components' in model.best_params_.keys(): 
                n_components = model.best_params_['reductor__lsa__n_components']
            if 'reductor__ipca__n_components' in model.best_params_.keys(): 
                n_components = model.best_params_['reductor__ipca__n_components']

            if 'classifier__n_estimators' in model.best_params_.keys(): 
                n_estimators = model.best_params_['classifier__n_estimators']
            if 'classifier__kernel' in model.best_params_.keys(): 
                kernel = model.best_params_['classifier__kernel']
            if 'classifier__n_neighbors' in model.best_params_.keys(): 
                n_neighbors = model.best_params_['classifier__n_neighbors'] 

            methods_pickle = pickle.dumps(model.estimator.steps)
            model_pickle = pickle.dumps(model)
        
        #ml_experience = {'date_experience':datetime.now(), 'type': 'O', 'methods_pickle': methods_pickle, 'code_experience' :code_experience,
        #       'score':score , 'matrice_confusion': matrice_confusion, 'autres_metriques' :  autres_metriques,
        #       'n_components': n_components, 'k': k, 'erreur_experience': erreur_experience, 'time_training': t_training , 'time_test' : t_test, 'model_pickle' : model_pickle}

            ml_experience = {'date_experience': datetime.now(), 'type': 'O', 'methods_pickle': methods_pickle, 'code_experience' :code_experience,
               'score':score , 'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp, 'erreur_model': erreur_model, 'fpr': fpr, 'fnr': fnr  , 'precision': precision  , 'rappel': rappel  , 
               'specificite': specificite, 'autres_metriques' :  autres_metriques,
               'n_components': n_components, 'k': k, 'n_estimators': n_estimators, 'kernel': kernel, 'n_neighbors': n_neighbors,
               'erreur_experience': erreur_experience, 'time_training': t_training , 'time_test' : t_test, 'model_pickle' : model_pickle}

            experience1_results.writelines(str(code_experience) + ';' + 'O' + ';' + str(score) + ';' + str(tn) + ';' + str(fp) + ';' + str(fn) + ';' + str(tp) + ';' + str(fpr) + ';' + str(fnr) + ';' + str(precision) + ';' + str(rappel) + ';' + str(specificite) + ';' + str(erreur_model) + ';' + str(k) + ';' + str(n_components)  + ';' +  str(kernel) + ';' + str(n_estimators) + ';' + str(n_neighbors) + ';' + str(t_training) + ';' + str(t_test) + '\n')
            experience1_results.close()

        #mongo_db = db_provider.MongoDB('experiences').ajouter_document(ml_experience)
        #return ml_experience

def construire_modele(X_train, y_train,  X_test, y_test, vectorizers, selectors, reductors, classifier):
    partition_cross_validation, parrallele_job = 3, 1
    pipe = Pipeline([('vectorizer', FeatureUnion(vectorizers)),
                        ('selector', FeatureUnion(selectors)),
                        ('reductor', FeatureUnion(reductors)),
                        ('classifier', classifier[1])])
    hyper_parameters = {**obtenir_hyper_parameters_selector_reductor(selectors + reductors), 
                        **obtenir_hyper_parameters_classifier(classifier[0])}
    
    return GridSearchCV(pipe, n_jobs=parrallele_job, param_grid=hyper_parameters, cv=partition_cross_validation).fit(X_train, y_train)

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

def obtenir_hyper_parameters_classifier(classifier):
    hyper_parameters = {}

    if classifier == 'rf':
        hyper_parameters = { 'classifier__n_estimators': [500, 750, 1000]}
    if classifier == 'svm':
        hyper_parameters = { 'classifier__kernel': ['linear', 'poly', 'rbf','sigmoid']}
    if classifier == 'kn':
        hyper_parameters = {'classifier__n_neighbors': [3, 5]}

    return hyper_parameters

def obtenir_hyper_parameters_selector_reductor(tous_les_selecteurs_reducteurs):
    hyper_parameters = {}
    composants, k =  [200, 250], [2000, 2500]

    for methode in tous_les_selecteurs_reducteurs:
        code_methode = methode[0]
        if code_methode == 'ki2':
            hyper_parameters = {**hyper_parameters, **{ 'selector__ki2__k': k}}
        if code_methode == 'mic':
            hyper_parameters = {**hyper_parameters, **{ 'selector__mic__k': k}}
        if code_methode == 'nmf':
            hyper_parameters = {**hyper_parameters, **{'reductor__nmf__n_components': composants}}
        if code_methode == 'lsa':
            hyper_parameters = {**hyper_parameters, **{'reductor__lsa__n_components': composants}}
        if code_methode == 'ipca':
            hyper_parameters = {**hyper_parameters, **{'reductor__ipca__n_components': composants}}

    return hyper_parameters

best_model_result = conduire_experience(sampling.echantilloner_donnees(),  definir_experiences())
#mongo_db = db_provider.MongoDB('experiences').ajouter_document(best_model_result)
d=1
