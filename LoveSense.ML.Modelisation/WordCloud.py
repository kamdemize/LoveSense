
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

def obtenir_nuage_mots(corpus):
    corpus_illegitime = corpus[corpus['alabel'] == 'illegitime']
    corpus_legitime = corpus[corpus['alabel'] == 'legitime']
    messages_corpus = {}
    messages_corpus['illegitime'] = ' '.join([mot for mot in corpus_illegitime['message']])
    messages_corpus['legitime'] = ' '.join([mot for mot in corpus_legitime['message']])

    stop_words = text.ENGLISH_STOP_WORDS
    wc = WordCloud(stopwords=stop_words, max_words=50,background_color="white", colormap="Dark2",
                   max_font_size=150, random_state=42)

    plt.rcParams['figure.figsize'] = [16, 6]
    classes = ['illegitime', 'legitime']
    for  index, classe in enumerate(messages_corpus):
        wc.generate(messages_corpus.get(classe))
    
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nuage de mots des messages de classe "+ classe)
        plt.show()


obtenir_nuage_mots(corpus.obtenir())
