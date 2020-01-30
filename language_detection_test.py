# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:52:48 2019

@author: DSKB5751
"""

from polyglot.text import Text, Word
import spacy
import treetaggerwrapper as ttpw
from langdetect import detect

#######################################Baseline detector#################################################################################
###############################################polyglot detector#########################################################################
liste=[]
for line in Data["Text"]:
    text=Text(" ".join(m for m in line))
    if text.language.code=="fr":
        liste.append("Fr")
    elif text.language.code=="en":
        liste.append("En")
    else:
        liste.append("other language")
Data["lang_predict_baseline"]=liste
liste_predit=[]
for i,row in Data.iterrows():
    if row["lang_predict_baseline"]=="En":
        liste_predit.append(0)
    if row["lang_predict_baseline"]=="Fr":
        liste_predit.append(1)
    if row["lang_predict_baseline"]=="other language":
        liste_predit.append(2)
liste=[]
for i,row in Data.iterrows():
    if row["lang"]=="En":
        liste.append(0)
    if row["lang"]=="Fr":
        liste.append(1)
###################################langdetect#############################################################################################

liste=[]
for line in Data["Text"]:
    if detect(" ".join(m for m in line))=="fr":
        liste.append("Fr")
    elif detect(" ".join(m for m in line))=="en":
        liste.append("En")
    else:
        liste.append("other language")
Data["lang_predict_baseline"]=liste
liste_predit=[]
for i,row in Data.iterrows():
    if row["lang_predict_baseline"]=="En":
        liste_predit.append(0)
    if row["lang_predict_baseline"]=="Fr":
        liste_predit.append(1)
    if row["lang_predict_baseline"]=="other language":
        liste_predit.append(2)
liste=[]
for i,row in Data.iterrows():
    if row["lang"]=="En":
        liste.append(0)
    if row["lang"]=="Fr":
        liste.append(1)
        
        
#######################################spacy###########################################
import spacy
from spacy_langdetect import LanguageDetector

liste=[]
for line in Data["Text"]:
    nlp = spacy.load("en")
    nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
    if nlp(" ".join(m for m in line))._.language["language"]=="fr":
        liste.append("Fr")
    elif nlp(" ".join(m for m in line))._.language["language"]=="en":
        liste.append("En")
    else:
        liste.append("other language")
Data["lang_predict_baseline"]=liste
liste_predit=[]
for i,row in Data.iterrows():
    if row["lang_predict_baseline"]=="En":
        liste_predit.append(0)
    if row["lang_predict_baseline"]=="Fr":
        liste_predit.append(1)
    if row["lang_predict_baseline"]=="other language":
        liste_predit.append(2)
liste=[]
for i,row in Data.iterrows():
    if row["lang"]=="En":
        liste.append(0)
    if row["lang"]=="Fr":
        liste.append(1)
