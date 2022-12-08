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
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
# RegEx for removing non-letter characters
import re
import random
import joblib,pickle
import os
import json


# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords # import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()
from roc_utils import *
 
from textblob import TextBlob
from textblob import Word
import string
from spellchecker import SpellChecker
from collections import Counter

def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
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

# spell = SpellChecker()

def b(x):
    spell = SpellChecker()
    text = x.lower()
#     text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
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

class radiologyretive(object):
    def __init__(self):
        self.LMmodel = SentenceTransformer('all-mpnet-base-v2')
        self.xgb_model = {}
        self.xgb_prediction=[]
        self.labels=['Daymaker sent']
        self.df_train=[]
        self.df_test=[]
        self.test_df = []
        self.X_train, self.X_test, self.y_train, self.y_test=[],[],[],[]

    #def getList(self, dict):
       # return list(dict.keys())
    # def save_json(self, filepath):
    #     json_txt = json.dumps(_saved_model, indent=4)
    #     with open(filepath, "w") as file:
    #         file.write(json_txt)

    # def load_json(self, filepath):
    #     with open(filepath, "r") as file:
    #         _saved_model = json.load(file)

    def model_load(self,modelpath):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(modelpath+'LMXgboost.json') 
        #self.xgb_model = pickle.load(open(modelpath+'LMXgboost.bin', 'rb'))
       
        #self.labels = self.getList(self.xgb_model)
        print('Model loaded!!')

        #return xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99)
        #try:
        #except:
        #    sys.exit('Model couldn\'t be loaded')

    def model_save(self, modelpath):
        try:
            self.xgb_model.save_model(modelpath+'LMXgboost.json')
             
            # pickle.dump(self.xgb_model, open(modelpath+'LMXgboost.json', 'wb'))
            print('Model saved!!')
        except:
            print('Model saving didn\'t worked')
        # try:
 
        #     pickle.dump(self.xgb_model, open(modelpath+'LMXgboost.json', 'wb'))
        #     print('Model saved!!')
        # except:
        #     print('Model saving didn\'t worked') 

        # self.labels = None
    def encode(self, sentences):
        return self.LMmodel.encode(sentences)

    def train_main(self, traindf, testdf):
        #load pre-trained model
        #traindf["Received_date"] = ""
        #traindf["Received_date"] = pd.to_datetime(traindf["Received_date"], format='%Y-%m-%d')
        #traindf = traindf.loc[(traindf["Received_date"] >= '2020-09-01')& (traindf["Received_date"] < '2020-09-15')] 

        self.df_train=traindf[['Comments','Daymaker sent']].dropna()
        self.df_test=testdf[['Comments','Daymaker sent']].dropna() 
        self.df_train = self.df_train.astype({'Comments':'string'})
        self.df_test = self.df_test.astype({'Comments':'string'})
        self.df_train  = self.df_train.reset_index(drop=True)
        self.df_test  = self.df_test.reset_index(drop=True)
        self.df_train['Cleaned_text']=self.df_train['Comments'].apply(b) 
        self.df_test['Cleaned_text']=self.df_test['Comments'].apply(b) 
        self.df_train['Comments_proc'] = self.df_train['Cleaned_text'].apply(review_to_words) 
        self.df_test['Comments_proc'] = self.df_test['Cleaned_text'].apply(review_to_words)  
  

        train_sentence_embeddings = self.encode(self.df_train['Comments_proc'])
        test_sentence_embeddings = self.encode(self.df_test['Comments_proc'])  

        y_train = self.df_train['Daymaker sent'].values
        y_valid = self.df_test['Daymaker sent'].values
        self.xgb_model = xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).fit(train_sentence_embeddings, y_train) 
        self.xgb_prediction = self.xgb_model.predict_proba(test_sentence_embeddings) 
        print(classification_report(y_valid, self.xgb_model.predict(test_sentence_embeddings)))

        # xgb_prediction = self.xgb_model[l].predict_proba(test_sentence_embeddings)
        # print(classification_report(y_valid, self.xgb_model[l].predict(test_sentence_embeddings)))
  
    
    def test_main(self, test,modelpath): 
        self.test_df=test[['Comments','Daymaker sent']].dropna()  
        self.test_df = self.test_df.astype({'Comments':'string'}) 
        self.test_df  = self.test_df.reset_index(drop=True)
        self.test_df['Cleaned_text']=self.test_df['Comments'].apply(b)
        self.test_df['Comments_proc'] = self.test_df['Cleaned_text'].apply(review_to_words) 

        pred_dyn = []
        test_sentence_embeddings = self.encode(self.test_df['Comments_proc']) 
        #rint('model:',self.model_load(modelpath))
        #model=self.model_load(modelpath).load_model(modelpath+"LMXgboost.json")
        #new
        # model=xgb.XGBClassifier(objective='binary:logistic', eta=0.3, silent=1, subsample=0.8, scale_pos_weight=99).load_model(modelpath+"LMXgboost.json")
        # self.xgb_prediction = model.predict_proba(test_sentence_embeddings) 
        self.xgb_prediction = self.xgb_model.predict_proba(test_sentence_embeddings) 
        pred_dyn.append(self.xgb_model.predict(test_sentence_embeddings))
        print('worked') 
        self.test_df['Daymaker sent'] = pred_dyn[0]
        return self.test_df 

    def get_y_and_heights(self,text_wrapped, dimensions, margin, font): 
        ascent, descent = font.getmetrics()
        line_heights = [
            font.getmask(text_line).getbbox()[3] + descent + margin
            for text_line in text_wrapped
        ] 
        line_heights[-1] -= margin 
        height_text = sum(line_heights) 
        y = (dimensions[1] - height_text) // 2 
        return (y, line_heights)

    def train_automate(self):
        arr = np.array(self.xgb_prediction)
        pred_df=pd.DataFrame(arr)
        comments = pd.DataFrame({"Comments":list(self.df_test['Cleaned_text'])})
        comments = comments["Comments"]
        pred_df = pred_df.join(comments)
        pred_df=pred_df.rename(columns={0: "Prediction_score"})
        pred_df_sort=pred_df.sort_values(by=["Prediction_score"], ascending=False)
        pred_df_sort.reset_index(inplace=True) 
        WIDTH = 1100
        HEIGHT = 400
        V_MARGIN =  10
        CHAR_LIMIT = 30
        BG_COLOR = "yellow"
        TEXT_COLOR = "black" 
        text = '" '+pred_df_sort['Comments'].iloc[0]+' "' 
        text = text.upper() 
        #font = ImageFont.truetype("/home/mnadella/ZakirahsBold.ttf", 35) 
        font = ImageFont.truetype("{0}/ZakirahsBold.ttf".format(os.getcwd()), 35) 
        img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR) 
        draw_interface = ImageDraw.Draw(img) 
        text_lines = wrap(text, CHAR_LIMIT) 

        y, line_heights = self.get_y_and_heights(
            text_lines,
            (WIDTH, HEIGHT),
            V_MARGIN,
            font
        )
        
        for i, line in enumerate(text_lines): 
            line_width = font.getmask(line).getbbox()[2]
            x = ((WIDTH - line_width) // 2) 
            draw_interface.text((x, y), line, font=font, fill=TEXT_COLOR) 
            y += line_heights[i] 
        # Opening the primary image (used in background)
        temp1 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template1.png".format(os.getcwd()))
        temp2 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template2.png".format(os.getcwd()))
        temp3 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template3.png".format(os.getcwd()))
        temp4 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template4.png".format(os.getcwd()))
        temp5 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template5.png".format(os.getcwd()))
        temp6 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template6.png".format(os.getcwd())) 
        test_list = [temp1,temp2,temp3,temp4,temp5,temp6]
        test_list 
        rand_idx = random.randrange(len(test_list))
        img1 = test_list[rand_idx]
        new_image = img1.resize((1920, 1280)) 
        img2 = img 
        new_image.paste(img2, (450, 530)) 
        # display(new_image)
        img_new = new_image.convert("RGBA")
        datas = img_new.getdata() 
        newData = []
        for item in datas:
            if item[1] > 225:
                newData.append((255, 255, 250, 0))
            else:
                newData.append(item)
        img_new.putdata(newData)

        img_new1 = img_new.convert("RGB") 
        # display(img_new1)
        #img_new1.save("/home/mnadella/RadiologyFeedback_meghana/new/img.jpg")
        img_new1.save("{0}/RadiologyFeedback_meghana/Daymakers/output/Template-Daymaker.png".format(os.getcwd()))


    def test_automate(self):
        arr = np.array(self.xgb_prediction)
        pred_df=pd.DataFrame(arr)
        comments = pd.DataFrame({"Comments":list(self.test_df['Cleaned_text'])})
        comments = comments["Comments"]
        pred_df = pred_df.join(comments)
        pred_df=pred_df.rename(columns={0: "Prediction_score"})
        pred_df_sort=pred_df.sort_values(by=["Prediction_score"], ascending=False)
        pred_df_sort.reset_index(inplace=True) 
        WIDTH = 1100
        HEIGHT = 400
        V_MARGIN =  10
        CHAR_LIMIT = 30
        BG_COLOR = "yellow"
        TEXT_COLOR = "black" 
        print(pred_df_sort['Comments'])
        text = '" '+pred_df_sort['Comments'].iloc[0]+' "' 
        text = text.upper() 
        #font = ImageFont.truetype("/home/mnadella/ZakirahsBold.ttf", 35) 
        font = ImageFont.truetype("{0}/ZakirahsBold.ttf".format(os.getcwd()), 35) 
        img = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR) 
        draw_interface = ImageDraw.Draw(img) 
        text_lines = wrap(text, CHAR_LIMIT) 

        y, line_heights = self.get_y_and_heights(
            text_lines,
            (WIDTH, HEIGHT),
            V_MARGIN,
            font
        )
        
        for i, line in enumerate(text_lines): 
            line_width = font.getmask(line).getbbox()[2]
            x = ((WIDTH - line_width) // 2) 
            draw_interface.text((x, y), line, font=font, fill=TEXT_COLOR) 
            y += line_heights[i] 
        # Opening the primary image (used in background)
        temp1 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template1.png".format(os.getcwd())) 
        temp2 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template2.png".format(os.getcwd()))
        temp3 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template3.png".format(os.getcwd()))
        temp4 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template4.png".format(os.getcwd()))
        temp5 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template5.png".format(os.getcwd()))
        temp6 = Image.open(r"{0}/RadiologyFeedback_meghana/Daymakers/templates/Template6.png".format(os.getcwd())) 
        test_list = [temp1,temp2,temp3,temp4,temp5,temp6]
        test_list 
        rand_idx = random.randrange(len(test_list))
        img1 = test_list[rand_idx]
        new_image = img1.resize((1920, 1280)) 
        img2 = img 
        new_image.paste(img2, (450, 530)) 
        # display(new_image)
        img_new = new_image.convert("RGBA")
        datas = img_new.getdata() 
        newData = []
        for item in datas:
            if item[1] > 225:
                newData.append((255, 255, 250, 0))
            else:
                newData.append(item)
        img_new.putdata(newData)

        img_new1 = img_new.convert("RGB") 
        # display(img_new1)
        #img_new1.save("/home/mnadella/RadiologyFeedback_meghana/new/img.jpg")
        img_new1.save("{0}/RadiologyFeedback_meghana/Daymakers/output/Template-Daymaker.png".format(os.getcwd()))



    









