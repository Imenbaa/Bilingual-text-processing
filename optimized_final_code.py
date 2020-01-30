# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:53:32 2019

@author: DSKB5751
"""

###############################Packages import#################################
import pandas as pd 
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import nltk
import treetaggerwrapper
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import operator
from functools import reduce
from polyglot.text import Text, Word
import numpy as np
import itertools
import treetaggerwrapper as ttpw
import ast
import confusion_matrix
from confusion_matrix import *
from sklearn import preprocessing
import collocations
from collocations import collocations,bigrams
import re
##############################Data Import######################################
Data=pd.read_csv("wikidata_sentences.csv")

def remove_newlines_unicode_english(fname):
    f=open(fname,"r",errors="ignore")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\t')[0].lower())
        f.close()
    return result
def remove_newlines_unicode_french(fname):
    f=open(fname,"r",errors="ignore",encoding="UTF-8")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split(' ')[0].lower())
        f.close()
    return result
def remove_newlines_unicode_utf(fname):
    f=open(fname,"r",errors="ignore",encoding="UTF-8")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\n')[0].lower())
        f.close()
    return result
def remove_newlines_unicode_iso(fname):
    f=open(fname,"r",errors="ignore",encoding="ISO-8859-1")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\n')[0].lower())
        f.close()
    return result
liste_en=remove_newlines_unicode_english("english.txt")
liste_fr=remove_newlines_unicode_french("french.txt") 
months_list=remove_newlines_unicode_iso("months-days.txt")
countries_list=remove_newlines_unicode_iso("countries.txt")
names_list=remove_newlines_unicode_utf("names.txt")
stp_fr=remove_newlines_unicode_utf("stop-words-fr.txt")
stp_en=remove_newlines_unicode_utf("stop-words-en.txt")
chiffre_latin=remove_newlines_unicode_utf("chiffre_latin.txt")
###########################Preprocessing and data cleaning#####################
characters = ["'","","-","Ø",".","®","$","»","§","«","�",'"',"÷","`","_","#",
              "@","?","!",";","[","]",":","Â°","(",")","&",",","|","/","â€™",
              "%","<",">","\\","+","â€”","*","=","~¤","¤~","—","’","€","$","»"
              ,"«","£","¥"
              ,'ℓ']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q",
           "r","s","t","u","v","w","x","y","z"]
'''
Clean and lowercase
'''
Data=Data.dropna()
Data=Data[Data["Text"].map(len)>2]
Data.reset_index(drop=True,inplace=True)
Data["Text"]=[line.lower() for line in Data["Text"]]

'''
   tokenize et supprimer les ref et les charactéres spéciaux
''' 
pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[a-z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
tokenizer=RegexpTokenizer(pattern)
for index,row in Data.iterrows():
    token=tokenizer.tokenize(str(row["Text"]))
    row["Text"]=[t for t in token if (t not in characters) & 
       (t.isdigit() == False) & (t not in chiffre_latin) & (t not in letters) &
       (t not in months_list) & (t not in  countries_list) &  (t not in names_list)]
'''
remove special formats
'''
aux5=[]
for line in Data["Text"]:
    aux2=[re.sub(r"\d+(?!g)\w+", "", str(w)) for w in line] ###### 1090s except 4g,3g,2g,5g
    aux3=[re.sub(r"\w+\d+", "", str(w)) for w in aux2] ########C3 
    aux4=[re.sub(r"\W", "", str(w)) for w in aux3]
    aux5.append([w for w in aux4 if w!=''])
Data["Text"]=aux5
liste_false=[57,567,568,597,1526]
for index,row in Data.iterrows():
    if index in liste_false:
        row["lang"]="En"
#Drop empty rows in Dataframe
Data=Data[Data["Text"].map(len)!=0]
Data.reset_index(drop=True,inplace=True)
#######################Language detection######################################
char_special_fr=["à","â","ç","é","è","œ","ê","ù","ô"]
my_liste_accent=[]
all_corpus=set(list(itertools.chain.from_iterable(Data["Text"])))
for w in all_corpus:
    for i in char_special_fr:
        if len(re.findall(i,w)):
            my_liste_accent.append(w)
def detection_langue(texte):
    if len(texte) > 2:
        words = set(texte)
        first_check = ["les","le","pas","ici","alors","aussi","encore","par","dans","sur","pour","avec","au","à","aux","après","avant"]
        commen_element_first_test = words.intersection(first_check)
        if len(commen_element_first_test) != 0: return('fr')
        common_elements_fr = set(words).intersection(liste_fr+my_liste_accent)
        common_elements_en = set(words).intersection(liste_en)
        if len(common_elements_fr.difference(common_elements_en)) >= len(common_elements_en.difference(common_elements_fr)) : return 'fr'
        if len(common_elements_en.difference(common_elements_fr)) > len(common_elements_fr.difference(common_elements_en)) : return "en"
        else: return "empty"

liste=[]
for line in Data["Text"]:
    if detection_langue(line)=="fr":
        liste.append("Fr")
    elif detection_langue(line)=="en":
        liste.append("En")
    else:
        liste.append("Empty")

Data["lang_predict"]=liste
Data=Data[Data.lang_predict != 'Empty']
Data.reset_index(drop=True,inplace=True)
#############Performance evaluation of language detection######################
le = preprocessing.LabelEncoder()
le.fit(Data["lang"])
True_liste=list(le.transform(Data["lang"]))
le.fit(Data["lang_predict"])
liste_predit=list(le.transform(Data["lang_predict"]))
'''
    Accuracy
'''
accuracy_score(True_liste, liste_predit)
accuracy_score(True_liste, liste_predit,normalize=False)
'''
    confusion matrix
'''
plot_confusion_matrix(True_liste, liste_predit, classes=["En","Fr"],
                      title='Confusion matrix, without normalization')

####################lemmatization##############################################
def pos_tag_new(texte):
    if texte!='':
        tagger = ttpw.TreeTagger(TAGLANG='en')
        tags = tagger.tag_text(texte)
        result=[t.split('\t')[1] for t in tags]
        for i,t in enumerate(result):
            if t[0] in ["V","N","J","R"]:
                    result[i]=t[0].lower()
            else:
                result[i]=""
    return result
def lemmatization_en(texte):
    lemma = list()
    lemmatizer = WordNetLemmatizer()
    if texte!='':
        couple=[(l,t) for l,t in zip(texte,pos_tag_new(texte)) if t!=""]
        lemma=[lemmatizer.lemmatize(w[0],pos=w[1]) for w in couple]
    return lemma

def lemmatization_fr(texte):
    lemma = list() 
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
    tags = tagger.tag_text(texte)
    tags2 = treetaggerwrapper.make_tags(tags)
    for row in range(0, len(tags2)):
        word = tags2[row][2]
        if word!='':
            wordPos = tags2[row][1]
            if wordPos[0] in ["N","A","V"]:
                if word.find("|") == -1: lemma.append(word)
                else: lemma.append(word.split("|")[0])
    return lemma
'''
    corpus intersection
'''
corpus_fr=[]
corpus_en=[]
for index,row in Data.iterrows():
    if row["lang_predict"]=="Fr":
        corpus_fr.append(row["Text"])
    if row["lang_predict"]=="En":
        corpus_en.append(row["Text"])
corpus_fr = list(itertools.chain.from_iterable(corpus_fr))
corpus_en = list(itertools.chain.from_iterable(corpus_en))
intersect=set(corpus_fr).intersection(set(corpus_en))
'''
    apply lemmatization functions
'''
liste_lemma=[]
for index,row in Data.iterrows():
    row["Text"]=[word for word in row["Text"] if word!='']
    if row["lang_predict"]=="Fr":
        if len(set(row["Text"]).intersection(intersect))==0:
            liste_lemma.append([lemmatization_fr(row["Text"])])
        else:
            a=[]
            for w in row["Text"]:
                a.append([lemmatization_en([w]) if w in intersect else
                          lemmatization_fr(w) for w in intersect])
            t=[]
            for w in a:
                t.append([word for word in w if word!=""]) 
            liste_lemma.append(t)
    else:
       liste_lemma.append(lemmatization_en(row["Text"]))
Data["lemma"]=liste_lemma
Data=Data[Data["lemma"].map(len)!=0]
Data.reset_index(drop=True,inplace=True)
'''
    delete list of list in lemma
'''
for index,row in enumerate(Data["lemma"]):
    if type(row[0])==list:
        Data["lemma"][index]=list(itertools.chain.from_iterable(row))
Data=Data[Data["lemma"].map(len)!=0]
Data.reset_index(drop=True,inplace=True)
##################Remove Stopwords############################################
'''
   Remove stopWords
'''
for i,row in Data.iterrows():
    if row["lang_predict"]=="Fr":
        row["lemma"]=[t for t in row["lemma"] if t not in stp_fr and len(t)>2]
    if row["lang_predict"]=="En":
        row["lemma"]=[t for t in row["lemma"] if t not in stp_en and len(t)>2]

Data=Data[Data["lemma"].map(len)!=0]
Data.reset_index(drop=True,inplace=True)
################Bigrams#######################################################

'''
    collocations from file
'''

f=open("bigrams.txt","r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(eval(x.split('\n')[0]))
f.close()
result=set(result)

def apply_collocations(listoflist_sentences,set_coloc):
    listoflist_sentences=' '.join(listoflist_sentences)
    for b1,b2 in set_coloc:
        listoflist_sentences = re.sub(r"(?<!_)%s %s(?!_)" % (b1 ,b2), "%s_%s" % (b1 ,b2),listoflist_sentences )
    return listoflist_sentences.split(" ")
for i,line in enumerate(Data["lemma"]):
    Data["lemma"][i]=apply_collocations(line,result)
############################Preprocessing result##############################
Data.to_csv("résultat_preprocessing.csv",index=False)



