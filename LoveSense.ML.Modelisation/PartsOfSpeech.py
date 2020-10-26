
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction import text
import spacy

nlp = spacy.load('en_core_web_sm')

#UserModule
import sys, os
from pathlib import Path
# @todo = refac later to the right way
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent)+'\\LoveSence.ML.Communs')
import AccessCorpus as corpus

nlp = spacy.load('en_core_web_sm')

def obtenir_nuage_parties_discours(corpus):
    corpus_illegitime = corpus[corpus['alabel'] == 'illegitime']
    corpus_legitime = corpus[corpus['alabel'] == 'legitime']

    messages_corpus = {}
    messages_corpus['illegitime'] = ' '.join([mot for mot in corpus_illegitime['message']])
    messages_corpus['legitime'] = ' '.join([mot for mot in corpus_legitime['message']])

    stop_words = text.ENGLISH_STOP_WORDS
    wc = WordCloud(stopwords=stop_words, max_words=50,background_color="white", colormap="Dark2",
                   max_font_size=150, random_state=42)

    classes = ['illegitime', 'legitime']
    for classe in ['illegitime', 'legitime']:
        tracer_nuage_mots(messages_corpus, classe)

def tracer_nuage_mots(messages_corpus, classe):
    entites_classe = obtenir_donnees_classe(messages_corpus, classe)
    etiquettes = [pdd for pdd in entites_classe.keys()]
    valeurs = [pdd for pdd in entites_classe.values()]
    plot_bar_x(etiquettes, valeurs, 'illégitime')  

def obtenir_donnees_classe(messages_corpus, classe):
    dico = {}
    messages_spacy = nlp(messages_corpus.get(classe))
    pos = [mot.pos_ for mot in messages_spacy]
    for p in pos:
        if (p in dico):
            dico[p] += 1
        else:
            dico[p] =  1        

    return {k: dico[k] for k in sorted(dico)}

def plot_bar_x(label, valeurs, classe):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, valeurs, color= 'blue')
    plt.xlabel('Partie du discours', fontsize=10)
    plt.ylabel("Nombre d'occurrence", fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title('Partie du discours des messages de classe ' + classe)
    plt.show()

obtenir_nuage_parties_discours(corpus.obtenir())