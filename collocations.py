# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:06:48 2019

@author: DSKB5751
"""

import nltk
import pandas as pd
import ast
from gensim.models import Phrases
from gensim.models.phrases import Phraser

df=pd.read_csv("last_data_version.csv")
liste=[]
for line in df["lemma"]:
    liste.append(ast.literal_eval(line))
def collocations(liste,occ):
    unlist_comments = [item for items in liste for item in items]
    #Collocations
    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(unlist_comments)
    #Counting frequencies
    bigram_freq = bigramFinder.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
    
    #PMI
    bigramFinder.apply_freq_filter(occ)
    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
    return bigramFreqTable,bigramPMITable
# gensim collocations bigrams and trigrams
def bigrams(list_of_list,occ,th):
    phrases = Phrases(list_of_list,min_count=occ,threshold=th)
    bigram = Phraser(phrases)
    for index,sentence in enumerate(list_of_list):
        list_of_list[index] = bigram[sentence]
    c=bigram.phrasegrams
    return list_of_list,c

#c is bigrams and trigrams

