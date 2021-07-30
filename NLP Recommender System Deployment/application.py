from pip._internal import main
import sys
import pandas as pd
import numpy as np
# import contractions
import nltk
import spacy
import en_core_web_md
nlp = en_core_web_md.load()
import string
punc = string.punctuation
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import flair
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
glove_embedding = WordEmbeddings('glove')
from flask import Flask,render_template,url_for,request,flash
import boto3
from io import StringIO
import awswrangler as wr
from gevent.pywsgi import WSGIServer

application=Flask(__name__)
application.secret_key = b'_5#ysfPfG"F4sfg8ez\n\xec]/'

from nltk.tokenize import word_tokenize
nltk.download('punkt')

STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

debug=True

#--------------------------------------------------------------------------------------------------------------------
# Config to s3
BUCKET = 'zappa-pqw2khami'
PATH_DIR = 'data/'

boto3_session = boto3.Session(
    aws_access_key_id="AKIASJE6COQ7W4FW75II",
    aws_secret_access_key="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",
    region_name="eu-west-2"
)

s3 = boto3_session.client('s3')

def read_csv_s3(filename):
    return pd.read_csv(s3.get_object(Bucket=BUCKET, Key=PATH_DIR+filename)['Body'], index_col=0, encoding = 'UTF-8')

def upload_csv_s3(df,filename):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Bucket=BUCKET, Key=PATH_DIR+filename, Body=csv_buffer.getvalue())
# --------------------------------------------------------------------------------------------------------------------
# Define functions

# Sys function
def printdebug(s):
    if debug==True:
        print(s)

# Data Pre-processing
def tokenisation(s):
    return word_tokenize(s.lower())

# def contract(s):
#     return list(map(lambda w: contractions.fix(w),s))

def puntuations(s):
    return [w for w in s if w not in punc]

def stopwords_rm(s):
    return [w for w in s if w not in STOP_WORDS]

def lemm(s):
    return [token.lemma_ for token  in nlp(' '.join(s))]

def pipe(s):
    return ' '.join(lemm(stopwords_rm(puntuations(tokenisation(s)))))

def apply_pipeline(df):
    cols = ['content_ps','content_re']
    for col in cols:
        df[col] = df[col].apply(lambda row: pipe(row))
    return df

def word_embedding(s):
    sentence = Sentence(s)
    glove_embedding.embed(sentence)
    sentence_matrix = sum([np.matrix(token.embedding) for token in sentence])/len(sentence)
    return np.array(sentence_matrix).ravel()

# Model training
def get_X_Y(df):
    cols = ['content_ps','content_re']
    for col in cols:
        df[col] = df[col].apply(word_embedding)
    ps = pd.DataFrame(df['content_ps'].to_list())
    re = pd.DataFrame(df['content_re'].to_list())

    X = pd.concat([ps,re],axis=1)
    y = df['label']
    
    return X, y

def get_model_rf(df):
    
    # Pre-processing
    df_rf = df.copy()
    ## Apply pipepline
    X, y = get_X_Y(apply_pipeline(df_rf))
    ## Split Training and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8, stratify=y)
    
    # Models
    ## Initial the model with the fine-tuned parameter
    rf_clf = RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=50, 
                                    max_features='sqrt', min_samples_leaf=2, min_samples_split=6, 
                                    n_estimators=550, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_train = rf_clf.predict(X_train)
    y_pred_test = rf_clf.predict(X_test)
    
    # Result
    ## Training Set
    printdebug('Train Set:')
    printdebug('Accuracy Score: {}'.format(accuracy_score(y_pred_train,y_train)))
    printdebug('F1 Score: {}'.format(f1_score(y_pred_train,y_train)))
    ## Test Set
    printdebug('Test Set:')
    printdebug('Accuracy Score: {}'.format(accuracy_score(y_pred_test,y_test)))
    printdebug('F1 Score: {}'.format(f1_score(y_pred_test,y_test)))
    
    return rf_clf

def process_new_ps(ps):
    return word_embedding(pipe(ps))

def predict_new_ps(ps,re,model):
    
    proba = []
    
    for i in range(10):
        new_x = np.append(process_new_ps(ps),re[i]).reshape(1,-1)
        # printdebug(models[i].predict_proba(new_x))
        proba.append(model.predict_proba(new_x)[0][1])
    
    recom = [i for i in range(10) if i in np.argsort(proba)[:-4:-1] and proba[i]>0.5]
    
    printdebug(proba)

    global recom_re
    recom_re = list(df_selected_re.iloc[recom].T.to_dict().values())
    
    return recom_re

# --------------------------------------------------------------------------------------------------------------------
# Model Training

# Get the raw data
df = read_csv_s3('final_data.csv')[['content_ps','title_re','content_re','label']]

# Get the resourcees and content
df_selected_re = read_csv_s3('selected_resources.csv')

resources_list = df_selected_re.iloc[:10,2].tolist()
content_re_list = df_selected_re.iloc[:10,3].tolist()

printdebug('Training Models...')
model = get_model_rf(df)

# For new PS
new_ps = ''
content_re_embedded = list(map(word_embedding,map(pipe,content_re_list)))
recom_re = []

# --------------------------------------------------------------------------------------------------------------------
# Interact with HTML

@application.route('/')
def home():
	return render_template('home.html')

@application.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        ps = request.form['message']
    
    global new_ps
    new_ps = ps

    return render_template('result.html',recom_re = predict_new_ps(new_ps,content_re_embedded,model), ps = new_ps)

@application.route('/feedback',methods=['POST'])
def feedback():
    responses = request.form.getlist('fb')

    # Check error
    if len(responses) == len(recom_re):
        # For Feedback
        try:
            df_new_data = read_csv_s3('new_data.csv')

        except:
            df_new_data = pd.DataFrame(columns=['content_ps', 'title_re', 'content_re','label'])
        
        new_data = [{'content_ps':new_ps,'title_re':recom_re[i]['title'],'content_re':recom_re[i]['text'],'label':responses[i]} for i in range(len(responses))]

        df_new_data = pd.concat([df_new_data,pd.DataFrame(new_data)],ignore_index=True)
        upload_csv_s3(df_new_data, 'new_data.csv')

        return render_template('feedback.html',ps=new_ps,error=False)
    else:
        return render_template('feedback.html',ps=new_ps,error=True)



if __name__ == '__main__':
    # application.run(debug=True)
    http_server = WSGIServer(('', 5000), application)
    http_server.serve_forever()