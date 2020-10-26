#NLP
import spacy
nlp = spacy.load('en_core_web_sm')
import re


def tokenize(texte):
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r"#(\w+)", '', texte)
    texte = re.sub(r"@(\w+)", '', texte)
    texte = re.sub(r'[^\w\s]', '', texte)
    texte = texte.strip().lower()
    return [str(x) for x in nlp(texte)] 

def clean(data):
    #tokennize
    data['clean_message'] = data['message'].map(lambda x: tokenize(x)).map(lambda tokens: ' '.join(tokens))
    data_clean = data[['alabel', 'clean_message']]
    
    # total par label/classe, just pour vérifier en debug
    a = data_clean['alabel'].value_counts()

    #vérification classe unique, just pour vérifier en debug
    b = data_clean['alabel'].unique()

    #supprimer données mnquantes (drop missing values)
    if (data_clean.isnull().sum()[1] != 0):
        corpus = data_clean.dropna()
     
    return data_clean