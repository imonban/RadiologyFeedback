from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
# RegEx for removing non-letter characters
import re
import csv
# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords
import sys
from nltk.stem.porter import *
import pickle

stemmer = PorterStemmer()
def review_to_words(review):
        # TODO: Remove HTML tags and non-letters,
        #soup = BeautifulSoup(review, 'html5lib')
        text = review.lower()
        #       convert to lowercase, tokenize,
        text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
        words = text.split()
        #       remove stopwords and stem
        words = [w.strip() for w in words if w not in stopwords.words('english')]
        words = [stemmer.stem(w) for w in words]

        # Return final list of words
        return ' '.join(words)




class radiologyretive(object):
    def __init__(self):
        self.LMmodel = SentenceTransformer('all-mpnet-base-v2')
        self.xgb_model = {}
        self.labels = None

    def encode(self, sentences):
        return self.LMmodel.encode(sentences)
    def train_main(self, traindf, testdf, labels):
    ##load pretarined model 
        self.labels =  labels
        try:
            traindf =traindf.fillna(' ')
            traindf['Comments_proc'] = traindf['Comments'].apply(review_to_words)
            testdf =testdf.fillna(' ')
            testdf['Comments_proc'] = testdf['Comments'].apply(review_to_words)
        except:
            sys.exit('Please provide a file with Comments')
        if any(item not in traindf.columns for item in labels) and any(item not in testdf.columns for item in labels):
            sys.exit('Please provide a correct labels in the train and test file')
        print('Training and validation file read')
        train_sentence_embeddings = self.encode(traindf['Comments'])
        test_sentence_embeddings = self.encode(testdf['Comments'])
        gt = []
        pred = []
        legend = []
        print(self.labels)
        for l in self.labels:
            try:
                y_train = traindf[l].values
                y_valid = testdf[l].values
                self.xgb_model[l] = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings, y_train) 
                xgb_prediction = self.xgb_model[l].predict_proba(test_sentence_embeddings)
                print(l)
                print(classification_report(y_valid, self.xgb_model[l].predict(test_sentence_embeddings)))
                gt.append(y_valid)
                pred.append(self.xgb_model[l].predict_proba(test_sentence_embeddings))
                legend.append(l)
                print('-------------------------------------------------')
            except:
                self.xgb_model[l]  =  'Null'
                print('Didnot work: '+l)

    def test_main(self, test):
        try:
            test =test.fillna(' ')
            test['Comments_proc'] = test['Comments'].apply(review_to_words)
        except:
            sys.exit('Please provide a file with Comments')
        pred_dyn = []
        test_sentence_embeddings = self.encode(test['Comments'])
        flg = []
        for l in self.labels:
            try:
                pred_dyn.append(self.xgb_model[l].predict(test_sentence_embeddings))
            except:
                flg.append(l)
                pred_dyn.append([0] * test_sentence_embeddings.shape[0])
                print('Didnot work: '+l)
        for i in range(len(self.labels)):
            if self.labels[i] not in flg:
                test[self.labels[i]] = pred_dyn[i]
        return test

    def getList(self, dict):
        return list(dict.keys())

        
    def model_load(self, modelpath):
        #try:
        self.xgb_model = pickle.load(open(modelpath+'LMXgboost.sav', 'rb'))
        self.labels = self.getList(self.xgb_model)
        print('Model loaded!!')
        #except:
        #    sys.exit('Model couldn\'t be loaded')

    def model_save(self, modelpath):
        try:
            pickle.dump(self.xgb_model, open(modelpath+'LMXgboost.sav', 'wb'))
            print('Model saved!!')
        except:
            print('Model saving didn\'t worked')
