import numpy as np
import pandas as pd

#to clean the tweets
import re
import nltk
#comment out the below two downloads after the first run
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.stem.porter import PorterStemmer

#sklearn suit
from sklearn.utils import resample #for upsampling the train dataset
from sklearn.feature_extraction.text import CountVectorizer # for extracting BOW features 
# from sklearn.feature_extraction.text import TfidfVectorizer # for extracting TDIDF features 

from sklearn.model_selection import train_test_split #for cross validation of train dataset
from sklearn.naive_bayes import GaussianNB #classifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import joblib # for saving our model
import pickle # for saving our model

def cleaning(traindf):
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
    return corpus

def model_start():
    #Read data files
    traindf = pd.read_csv('data/train.csv')
    # print (traindf)
    #drop duplicates
    traindf.drop_duplicates(inplace=True)

    #clean traindf tweets
    corpus = cleaning(traindf)
    #assign corpus to df and rename column
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
    bow_cv = CountVectorizer(max_features=1000).fit(train_upsampled['cleaned'])
    #save vectorizer
    vectorizer = 'vectorizer.pkl'
    joblib.dump(bow_cv, vectorizer)
    #transform to dense array
    counter = joblib.load(vectorizer)
    X = counter.transform(train_upsampled['cleaned']).toarray()
    y = train_upsampled['label']

    #Split train dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
    #Build the model
    nbbow = GaussianNB().fit(X_train, y_train) 
    prediction = nbbow.predict(X_val)
    #print scores
    print(f"F1 score : {f1_score(y_val, prediction)}")
    print(f"Training Data Score: {nbbow.score(X_train, y_train)}")
    print(f"Validation Data Score: {nbbow.score(X_val, y_val)}")
    print(classification_report(y_val, prediction))
    
    #save model
    model = 'classifier.pkl'
    joblib.dump(nbbow, open(model, 'wb'))

# model_start()

def tweet_predict(input1):
    #load model
    classifier = joblib.load(open('classifier.pkl', "rb"))
    vectorizermodel = joblib.load(open('vectorizer.pkl', "rb"))
    #meake a datframe of the input
    traindf = pd.DataFrame(columns=['tweet'])
    traindf = traindf.append({"tweet": input1['example-form']}, ignore_index=True)
    print(input1)
    #clean the tweet
    corpus = cleaning(traindf)
    print(corpus)
    traindf['cleaned'] = np.array(corpus)
    traindf = traindf.drop(columns=['tweet'])
    print(traindf)
    #vectorized the cleaned tweet
    vectorized_input = vectorizermodel.transform(traindf['cleaned']).toarray()
    print(vectorized_input)

    #classify the tweet
    result = classifier.predict(vectorized_input)
    print(result)

    return result[0]

