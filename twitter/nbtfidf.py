import numpy as np
import pandas as pd

#to clean the tweets
import re
import nltk
#comment out these two after the first run
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.stem.porter import PorterStemmer

#sklearn suit
from sklearn.utils import resample #for upsampling the train dataset
from sklearn.feature_extraction.text import CountVectorizer # for extracting BOW features 
from sklearn.feature_extraction.text import TfidfVectorizer # for extracting TDIDF features 

from sklearn.model_selection import train_test_split #for cross validation of train dataset
from sklearn.naive_bayes import GaussianNB #classifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import joblib # for saving our model
import pickle # for saving our model

def model_start():

    #Read data files
    traindf = pd.read_csv('data/train.csv')
    # print (traindf)
    testdf = pd.read_csv('data/test.csv')
    #drop duplicates
    traindf.drop_duplicates(inplace=True)
    testdf.drop_duplicates(inplace=True)
    #clean traindf tweets
    corpus = []
    for i in range (0, len(traindf)):
        tweet = traindf['tweet'][i]
        tweet = tweet.lower()
        tweet = re.sub(r'[^a-zA-Z]', ' ', tweet) #only alphabet
        tweet = re.sub(r'((www\.[^/s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub(r'@[^\s]+', 'AT_USER',  tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = tweet.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')+ list(punctuation) + ['AT_USER','URL', 'user']
        tweet = [ps.stem(word) for word in tweet if not word in set(all_stopwords)]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
        
    #assign corpus to each df and rename columns
    traindf['cleaned'] = np.array(corpus)
    train = traindf.drop(columns=['id', 'tweet'])
    
    #upsampling minority class of Train dataset
    #rename dfs with majority(non_hate) and nimority(hate)
    train_majority = train[train['label'] == 0]
    train_minority = train[train['label'] ==1]
    #Upsample minority
    train_minority_upsampled = resample (train_minority, replace=True, #sample with replacement
                                     n_samples=len(train_majority),# to match majority class
                                     random_state=42) # reproducible results 
    #Concatanate train_minority_upsampled to train_majority
    train_upsampled = pd.concat([train_minority_upsampled, train_majority])
    #Extract Features
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    train_upsampled_tfidf = tfidf_vectorizer.fit_transform(train_upsampled['cleaned']).toarray()
    #Split train dataset
    X_train, X_val, y_train, y_val = train_test_split(train_upsampled_tfidf, train_upsampled['label'], test_size = 0.2, random_state = 42)
 
    #Build the model
    nbtfidf = GaussianNB().fit(X_train, y_train) 
    prediction = nbtfidf.predict(X_val)
    #print scores
    print(f"F1 score : {f1_score(y_val, prediction)}")
    print(f"Training Data Score: {nbtfidf.score(X_train, y_train)}")
    print(f"Validation Data Score: {nbtfidf.score(X_val, y_val)}")
    print(classification_report(y_val, prediction))
    
    #save model
    model = 'nbtfidf.pkl'
    joblib.dump(nbtfidf, open(model, 'wb'))

    return

model_start()

def tweet_predict (input1):
    #load model
    model = joblib.load(open('nbtfidf.pkl', "rb"))
    output = model.predict(input1.values())

    return output[0]



