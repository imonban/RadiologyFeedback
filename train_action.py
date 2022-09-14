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




class radiologyaction(object):
    def __init__(self):
        self.LMmodel = SentenceTransformer('all-mpnet-base-v2')
        self.xgb_model = None
        self.labels = None
    def num_conv(txt):
        return label_dict[txt]
        
    def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
        classes = np.unique(y_train, axis = 0)
        classes.sort()
        class_samples = np.bincount(y_train)
        total_samples = class_samples.sum()
        n_classes = len(class_samples)
        weights = total_samples / (n_classes * class_samples * 1.0)
        class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
        class_weight_dict[classes[1]] = class_weight_dict[classes[1]]*largest_class_weight_coef
        sample_weights = [class_weight_dict[y] for y in y_train]
        return sample_weights
    def encode(self, sentences):
        return self.LMmodel.encode(sentences)
    def train_main(self, traindf, testdf, labels):
    ##load pretarined model 
        self.labels =  labels
        try:
            traindf =traindf.fillna(' ')
            traindf['Comments_proc'] = traindf['Comments'].apply(review_to_words)
            traindf = traindf.fillna('null')
            traindf = traindf[traindf[ 'Actions Annotated']!='null']
            testdf =testdf.fillna(' ')
            testdf['Comments_proc'] = testdf['Comments'].apply(review_to_words)
        except:
            sys.exit('Please provide a file with Comments')
        if any(item not in traindf.columns for item in labels) and any(item not in testdf.columns for item in labels):
            sys.exit('Please provide a correct labels in the train and test file')
        print('Training and validation file read')
        train_sentence_embeddings = self.encode(traindf['Comments'])
        test_sentence_embeddings = self.encode(testdf['Comments'])
        label_dict = {'Praise':0, 'Contacted Supervisor':1, 'Patient Feedback':2,'Direct Staff Feedback':3, 'Process Improvement':4, 'Nothing can be Done':5, 'Human Error':6, 'Technology Improvement':7}
        
        traindf['ActionsClass'] =  traindf['Actions Annotated'].apply(num_conv)
        testdf['ActionsClass'] =  testdf['Actions Annotated'].apply(num_conv)
        y_train = traindf['ActionsClass'].values
        y_valid = testdf['ActionsClass'].values
        #pass y_train as numpy array 
        weight = CreateBalancedSampleWeights(np.asarray(y_train), 0.91)
        gt = []
        pred = []
        legend = []
        print(self.labels)
        xgb_model = xgb.XGBClassifier( eta=0.3, silent=1, subsample=0.8, weights = weight, objective='multi:softmax').fit(train_sentence_embeddings, y_train) 
        xgb_model.predict(test_sentence_embeddings)

    def test_main(self, test):
        try:
            test =test.fillna(' ')
            test['Comments_proc'] = test['Comments'].apply(review_to_words)
        except:
            sys.exit('Please provide a file with Comments')
        test_sentence_embeddings = self.encode(test['Comments'])
        flg = []
        try:
            pred_dyn = self.xgb_model.predict(test_sentence_embeddings)
        except:
            flg.append(l)
            pred_dyn = [0] * test_sentence_embeddings.shape[0]
            print('Didnot work: ')
        la_dict = {0:'Praise', 1:'Contacted Supervisor', 2:'Patient Feedback',3:'Direct Staff Feedback', 4:'Process Improvement', 5:'Nothing can be Done', 6:'Human Error', 7:'Technology Improvement'}
        action_label = []
        for i in pred_dyn:
            action_label.append(la_dict[i])
        test['Actions'] = action_label
        return test


        
    def model_load(self, modelpath):
        #try:
        self.xgb_model = pickle.load(open(modelpath+'LMXgboost_action.sav', 'rb'))
        print('Model loaded!!')
        #except:
        #    sys.exit('Model couldn\'t be loaded')

    def model_save(self, modelpath):
        try:
            pickle.dump(self.xgb_model, open(modelpath+'LMXgboost_action.sav', 'wb'))
            print('Model saved!!')
        except:
            print('Model saving didn\'t worked')
