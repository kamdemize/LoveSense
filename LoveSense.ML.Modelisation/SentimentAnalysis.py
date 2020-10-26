import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import pylab

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import AccessCorpus as corpus
 
def affiche_dispersion_sentiments(corpus):
    plt.rcParams['figure.figsize'] = [25, 15]
    for index, row in corpus.iterrows():
        x = row.polarite
        y = row.subjectivite
        label = row.alabel
        if label == 'legitime':
            plt.scatter(x, y,  c='blue')
        else:
            plt.scatter(x, y,  c='red')
        #plt.xlim(-.01, .12) 
    
    plt.title('Analyse de sentiments', fontsize=20)
    plt.xlabel('<-- Negatif -------- Positif -->', fontsize=15)
    plt.ylabel('<-- Objectif -------- Subjectif -->', fontsize=15)
    plt.show()

def affiche_statistiques_descriptives(corpus, titre):
    print(titre)
    print('------------------------------------')
    print(corpus[corpus.alabel.eq('legitime')].shape[0], ' messages légitimes')
    print(corpus[corpus.alabel.eq('illegitime')].shape[0], ' messages légitimes')

    print('------')
    print('moyenne polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.mean())
    print('moyenne polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.mean())

    print('moyenne subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.mean())
    print('moyenne subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.mean())
    print('------')
    print('médiane polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.median())
    print('médiane polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.median())

    print('médiane subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.median())
    print('médiane subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.median())
    print('------')
    print('mode polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.mode())
    print('mode polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.mode())

    print('mode subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.mode())
    print('mode subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.mode())
    print('------')
    print('etendu polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.max() - corpus[corpus.alabel.eq('legitime')].polarite.min())
    print('etendu polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.max() - corpus[corpus.alabel.eq('illegitime')].polarite.min())

    print('etendu subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.max() - corpus[corpus.alabel.eq('legitime')].subjectivite.min())
    print('etendu subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.max() - corpus[corpus.alabel.eq('illegitime')].subjectivite.min())
    print('------')
    print('variance polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.var())
    print('movariancede polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.var())

    print('variance subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.var())
    print('variance subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.var())
    print('------')
    print('écart type polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.std())
    print('écart type polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.std())

    print('écart type subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.std())
    print('écart type subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.std())
    print('------')
    print('quantile Q1 polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.quantile(0.25))
    print('quantile Q1 polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.quantile(0.25))

    print('quantile Q1 subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.quantile(0.25))
    print('quantile Q1 subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.quantile(0.25))
    print('------')
    print('quantile Q2 polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.quantile(0.5))
    print('quantile Q2 polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.quantile(0.5))

    print('quantile Q2 subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.quantile(0.5))
    print('quantile Q2 subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.quantile(0.5))
    print('------')
    print('quantile Q3 polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.quantile(0.75))
    print('quantile Q3 polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.quantile(0.75))

    print('quantile Q3 subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.quantile(0.75))
    print('quantile Q3 subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.quantile(0.75))

    print('------')
    print('Coefficient de variation (écart-type / moyenne) *100 polarité legitime : ', (corpus[corpus.alabel.eq('legitime')].polarite.std()/ corpus[corpus.alabel.eq('legitime')].polarite.mean())*100)
    print('Coefficient de variation (écart-type / moyenne) *100 polarité illegitime : ',(corpus[corpus.alabel.eq('illegitime')].polarite.std()/corpus[corpus.alabel.eq('illegitime')].polarite.mean())*100)

    print('Coefficient de variation (écart-type / moyenne) *100 subjectivité legitime : ', (corpus[corpus.alabel.eq('legitime')].subjectivite.std()/corpus[corpus.alabel.eq('legitime')].subjectivite.mean())*100)
    print('Coefficient de variation (écart-type / moyenne) *100 subjectivité illegitime : ', (corpus[corpus.alabel.eq('illegitime')].subjectivite.std()/corpus[corpus.alabel.eq('illegitime')].subjectivite.mean())*100)

    print(corpus[corpus.alabel.eq('legitime')].polarite.min())
    print('etendu polarité legitime : ', corpus[corpus.alabel.eq('legitime')].polarite.max())

    print(corpus[corpus.alabel.eq('illegitime')].polarite.min())
    print('etendu polarité illegitime : ', corpus[corpus.alabel.eq('illegitime')].polarite.max()  )

    print(corpus[corpus.alabel.eq('legitime')].subjectivite.min())
    print('etendu subjectivité legitime : ', corpus[corpus.alabel.eq('legitime')].subjectivite.max()  )

    print(corpus[corpus.alabel.eq('illegitime')].subjectivite.min())
    print('etendu subjectivité illegitime : ', corpus[corpus.alabel.eq('illegitime')].subjectivite.max() )

def affiche_valeurs_abberantes(corpus, titre):
    print(titre)
    print('------------------------------------')
    valeur_max =3*corpus[corpus.alabel.eq('legitime')].polarite.std()
    abbereances =corpus[~corpus.alabel.eq('legitime') & abs(corpus.polarite) > valeur_max].polarite
    total = corpus.shape[0]
    abberantes = abbereances.shape[0]
    taux = (abberantes/total)*100
    print(f'polarite/legitime, sur {total} messages, {abberantes} sont abberantes, pour un taux de {str(taux)}')
 
    valeur_max =3*corpus[corpus.alabel.eq('illegitime')].polarite.std()
    abbereances =corpus[~corpus.alabel.eq('illegitime') & abs(corpus.polarite) > valeur_max].polarite
    total = corpus.shape[0]
    abberantes = abbereances.shape[0]
    taux = (abberantes/total)*100
    print(f'polarite/illegitime, sur {total} messages, {abberantes} sont abberantes, pour un taux de {str(taux)}')

    valeur_max =3*corpus[corpus.alabel.eq('legitime')].subjectivite.std()
    abbereances =corpus[~corpus.alabel.eq('legitime') & abs(corpus.subjectivite) > valeur_max].subjectivite
    total = corpus.shape[0]
    abberantes = abbereances.shape[0]
    taux = (abberantes/total)*100
    print(f'subjectivite/legitime, sur {total} messages, {abberantes} sont abberantes, pour un taux de {str(taux)}')

    valeur_max =3*corpus[corpus.alabel.eq('illegitime')].subjectivite.std()
    abbereances =corpus[~corpus.alabel.eq('illegitime') & abs(corpus.subjectivite) > valeur_max].subjectivite
    total = corpus.shape[0]
    abberantes = abbereances.shape[0]
    taux = (abberantes/total)*100
    print(f'subjectivite/illegitime, sur {total} messages, {abberantes} sont abberantes, pour un taux de {str(taux)}')

def affiche_boite_moustache(corpus):
    BoxName = ['legitime','illegitime']
    data = [corpus[corpus.alabel.eq('legitime')].polarite,corpus[corpus.alabel.eq('illegitime')].polarite]
    plt.boxplot(data ,0, 'gD')
    plt.ylim(-1,1)
    pylab.xticks([1,2], BoxName)
    plt.title('Boîte à mostache de la polarité')
    #plt.savefig('polarite_moustache.png')
    plt.show()

    BoxName = ['legitime','illegitime']
    data = [corpus[corpus.alabel.eq('legitime')].subjectivite,corpus[corpus.alabel.eq('illegitime')].subjectivite]
    plt.boxplot(data,1, 'gD')
    plt.ylim(0,1)
    pylab.xticks([1,2], BoxName)
    plt.title('Boîte à mostache de la subjectivité')
    #plt.savefig('subjectivite_moustache.png')
    plt.show()
    
def corpus_with_sentiments(corpus):
    polarite = lambda x: TextBlob(x).sentiment.polarity
    subjectivite = lambda x: TextBlob(x).sentiment.subjectivity
    corpus['polarite'] = corpus['message'].apply(polarite)
    corpus['subjectivite'] = corpus['message'].apply(subjectivite)

    return corpus

the_sentiment_corpus = corpus_with_sentiments(corpus.obtenir())
affiche_dispersion_sentiments(the_sentiment_corpus)
affiche_boite_moustache(the_sentiment_corpus)
affiche_valeurs_abberantes(the_sentiment_corpus, "Determination des valeurs aberrantes")
affiche_statistiques_descriptives(the_sentiment_corpus, "Statistiques descriptives")