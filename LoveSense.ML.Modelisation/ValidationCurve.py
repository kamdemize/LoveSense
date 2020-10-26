# sources inpirés de scikit-learn.org
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

import matplotlib.pyplot as plt
import numpy as np

#ML.Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve

#UserModule
import DatasSampling as echantillonage

def tracer_courbe_validation(donnees_entrainement, classifieur, parametre, plage_parametre, partition):
    train_scores, test_scores = validation_curve(classifieur, donnees_entrainement[0], donnees_entrainement[1], param_name=parametre, param_range=plage_parametre,
        scoring="accuracy", n_jobs=-1, cv=partition)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Courbe de validation du classeur")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="score d'entraînement",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="score de validation croisée",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


echantillon_donnees = echantillonage.obtenir_donnees_entrainement()
tracer_courbe_validation((echantillon_donnees[0], echantillon_donnees[1]),
                         SVC(),
                         'n_estimators',
                         [100, 200],
                         3)