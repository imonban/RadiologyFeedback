import pandas as pd
import numpy as np
import seaborn as sns
import string
import xgboost as xgb  
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  
import re
import random
import joblib,pickle
import os
import json  
import nltk
nltk.download("stopwords")   
from nltk.corpus import stopwords  

from nltk.stem.porter import *
from spellchecker import SpellChecker
spell = SpellChecker()

stemmer = PorterStemmer()
from roc_utils import *
 
from textblob import TextBlob
from textblob import Word
import string 
from collections import Counter

header = "/home/mnadella/RadiologyFeedback_meghana/" 
labels = ['Nursing', 'Modality Supervisor','Scheduling','Outside radiology','Radiologist MD','garbage_class']
y_map={0:'Nursing',1:'Modality Supervisor',2:'Scheduling',3:'Outside radiology',4:'Radiologist MD',5:'garbage_class'}

def review_to_words(x):
    text = x.lower() 
    exclist = string.punctuation
    # remove punctuations from oldtext
    table_ = str.maketrans('', '', exclist)
    text = text.translate(table_)
    words = text.split()
    y=[] 
    for word in words:
        y.append(spell.correction(word))
    try:
        if None in y:
            y[y.index(None)]=words[y.index(None)]
        return ' '.join(y)
    except:
        return ' '.join(words)

# spell = SpellChecker()


class radiologyretive(object):
    def __init__(self):
        self.LMmodel = SentenceTransformer('bert-base-cased-finetuned-mrpc') 
        self.model = {}
        self.model1 = {}
        self.xgb_prediction=[]
        self.xgb_predictiontest=[]
        self.labels=['Daymaker sent']
        self.df_train=[]
        self.df_test=[]
        self.df_pred=[]
        self.df_predtest=[]
        self.test_df = []
        self.X_train, self.X_test, self.y_train, self.y_test=[],[],[],[]

    def model_load(self,modelpath):
        k=0
        for i in labels:
            self.model1[k] = xgb.XGBClassifier()
            self.model1[k].load_model(modelpath+i+'LMXgboost.json')  
            print('Model loaded!!')
            k=k+1
 

    def model_save(self, modelpath):
        try:
            k=0
            for i in labels: 
                self.model[k].save_model(modelpath+i+'LMXgboost.json')
                k=k+1
            print('Model saved!!') 
        except:
            print('Model saving didn\'t worked')  

    

    def encode(self, sentences):
        return self.LMmodel.encode(sentences)

    def train_main(self, traindf, testdf): 
        traindf.rename(columns = {'DISAPPOINTED_COMMENTS':'Comments'}, inplace = True)
        traindf = traindf[labels[0:]+ ['Comments']] 
        testdf.rename(columns = {'DISAPPOINTED_COMMENTS':'Comments'}, inplace = True)
        testdf = testdf[labels[0:]+ ['Comments']] 
        traindf = traindf.astype({'Comments':'string'})
        testdf = testdf.astype({'Comments':'string'})
        traindf  = traindf.reset_index(drop=True)
        testdf  = testdf.reset_index(drop=True)

        self.df_train=traindf.dropna()
        self.df_test=testdf.dropna() 
        
        self.df_train['True_label']=self.df_train[labels[0:]].values.argmax(axis=1)
        self.df_test['True_label']=self.df_test[labels[0:]].values.argmax(axis=1)

        self.df_train['Cleaned_text'] = self.df_train['Comments'].apply(review_to_words) 
        self.df_test['Cleaned_text'] = self.df_test['Comments'].apply(review_to_words)  

        self.df_train = self.df_train[self.df_train['Cleaned_text'].str.split().str.len().gt(5)]
        self.df_train  = self.df_train.reset_index(drop=True) 
        self.df_test = self.df_test[self.df_test['Cleaned_text'].str.split().str.len().gt(5)]
        self.df_test  = self.df_test.reset_index(drop=True)

        self.df_train['classes']=self.df_train['True_label'].map(y_map)
        self.df_test['classes']=self.df_test['True_label'].map(y_map) 

        train_sentence_embeddings = self.encode(self.df_train['Cleaned_text'])
        test_sentence_embeddings = self.encode(self.df_test['Cleaned_text'])  

        # y_train = self.df_train['True_label'].values
        # y_valid = self.df_test['True_label'].values 

        j=0
        #results_df = pd.DataFrame({"Comments": list(X_test), "True_label": test.True_label, "Class":(test.True_label).map(y_map)})
        for i in labels: 
            y_train = self.df_train[i].values
            y_valid = self.df_test[i].values
            scale_pos_weight = self.df_train[self.df_train[i]==0].shape[0]/self.df_train[self.df_train[i]==1].shape[0]
            self.model[j] = xgb.XGBClassifier(objective='binary:logistic', eta= 0.3, silent=1, sub_sample=0.8, n_estimators =500, min_child_weight = 3, max_depth= 4,scale_pos_weight=scale_pos_weight).fit(train_sentence_embeddings, y_train)
            y_pred = self.model[j].predict(test_sentence_embeddings)
            self.xgb_prediction = self.model[j].predict_proba(test_sentence_embeddings)
            print("--------------------",i,"------------------------") 
            print(classification_report(y_valid, y_pred))
            j=j+1
            cm = confusion_matrix(y_valid, y_pred) 
            print(cm)   
            self.df_pred = pd.DataFrame({"Comments":list(self.df_test['Cleaned_text']), "True_label": self.df_test.True_label, "Class":(self.df_test.True_label).map(y_map)})
            self.df_pred[i] = y_valid
            self.df_pred["Pred_"+i] = y_pred 

        self.df_pred = self.df_pred.reset_index(drop=True)
        self.df_pred.to_excel(header+"Teams/output/pred.xlsx")

 
    
    def test_main(self, test,modelpath):  
        test.rename(columns = {'DISAPPOINTED_COMMENTS':'Comments'}, inplace = True)
        test = test[labels[0:]+ ['Comments']]  
        test = test.astype({'Comments':'string'}) 
        test  = test.reset_index(drop=True) 
        self.test_df=test.dropna() 
         
        self.test_df['True_label']=self.test_df[labels[0:]].values.argmax(axis=1)
 
        self.test_df['Cleaned_text'] = self.test_df['Comments'].apply(review_to_words)  
 
        self.test_df = self.test_df[self.test_df['Cleaned_text'].str.split().str.len().gt(5)]
        self.test_df  = self.test_df.reset_index(drop=True)
 
        self.test_df['classes']=self.test_df['True_label'].map(y_map) 
 
        test_sentence_embeddings = self.encode(self.test_df['Cleaned_text'])

        self.df_pred = pd.DataFrame({"Comments":list(self.test_df['Cleaned_text']), "True_label": self.test_df.True_label, "Class":(self.test_df.True_label).map(y_map)})
        # self.df_pred[i] = y_valid
        # self.df_pred["Pred_"+i] = y_pred 
        for i in range(len(labels)):
            #self.xgb_predictiontest[i] = self.model1[i].predict_proba(test_sentence_embeddings) 
            print(self.model1[i])
            y_pred = self.model1[i].predict(test_sentence_embeddings) 
            self.df_pred["Pred_"+labels[i]] = y_pred

        self.df_pred = self.df_pred.reset_index(drop=True)
        self.df_pred.to_excel(header+"Teams/output/predTest.xlsx") 
        print('worked')  
        return self.df_pred 

     

     