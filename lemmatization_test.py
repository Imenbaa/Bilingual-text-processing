# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:49:25 2019

@author: DSKB5751
"""
import nltk
import treetaggerwrapper
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import gensim
from gensim.utils import lemmatize
import treetaggerwrapper as ttpw
import spacy 
####################################################lemmatization################################################################
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0]
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def lemmatization_en(texte,langue):
    lemma = list()
    if langue =="En":
        # 1. Init Lemmatizer
        lemmatizer = WordNetLemmatizer()
        if get_wordnet_pos(texte)!='r' and get_wordnet_pos(texte)!="j" and texte!='':
            # 2. Lemmatize Single Word with the appropriate POS tag
            lemma.append(lemmatizer.lemmatize(texte, get_wordnet_pos(texte)))
    return lemma,get_wordnet_pos(texte)
[lemmatization_en(w,"En") for w in ["the","best","moment","was","when","I","saw","my","babies","walking","on","their","feet","correctly"]]
###################spacy lemmatization################################################
nlp = spacy.load('en', disable=['parser', 'ner'])
sentence = " ".join(["lovely","loving","lovers","hurting"])
doc = nlp(sentence)
[(token.lemma_,token.pos_) for token in doc]
###################spacy lemmatization################################################
nlp = spacy.load('fr', disable=['parser', 'ner'])
sentence = " ".join(["j","ai","codé","python","j","ai","des","spécialités", "en","data","et","réseaux","neurones"])
doc = nlp(sentence)
[(token.lemma_,token.pos_) for token in doc]
###################gensim lemmatization################################################
sentence = " ".join(["the","best","moment","was","when","I","saw","my","babies","walking","on","their","feet","correctly"])
[(wd.decode('utf-8').split('/')[0],wd.decode('utf-8').split('/')[1]) for wd in lemmatize(sentence)]
###################gensim lemmatization################################################
sentence = " ".join(["j","ai","codé","python","j","ai","des","spécialités", "en","data","et","réseaux","neurones"])
[(wd.decode('utf-8').split('/')[0],wd.decode('utf-8').split('/')[1]) for wd in lemmatize(sentence)]
###################treetagger lemmatization################################################
tagger = ttpw.TreeTagger(TAGLANG='en')
tags = tagger.tag_text(" ".join(["big data"]))
[(t.split('\t')[-1],t.split('\t')[1]) for t in tags]
###################treetagger lemmatization################################################
tagger = ttpw.TreeTagger(TAGLANG='en')
tags = tagger.tag_text("I am doing this correctly ")
[(t.split('\t')[-1],t.split('\t')[1]) for t in tags]
#[t.split('\t')[1] for t in tags]
