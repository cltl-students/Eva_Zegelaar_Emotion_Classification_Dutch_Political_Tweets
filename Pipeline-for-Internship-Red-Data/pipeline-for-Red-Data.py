import pandas as pd
import numpy as np 
from sklearn import svm #model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #bag of words vectorization
import gensim
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import csv
import re
import pandas as pd
import numpy as np 
from langdetect import detect
import sys
import json


newtweets = sys.argv[1]

#------------------------- PREPROCESSING OF RAW TWEETS---------------------------------------
## THE RAW TWEETS SHOULD BE A CSV FILE CONTAINING COLUMNS IN THE FOLLOWING ORDER: 'AUTEUR', 'DATUM', 'PARTIJ', CONTENT'. 
def read_csv(newtweets, encoding='utf-8'):
    """
    Reads a csv file using and splits each row into lists.
    (Make sure the csv file contains all the correct columns and rows)
    """
    authorlist = []
    datelist = []
    partylist = []
    contentlist = []
    list_dict = []
    contentlist = []
    with open(newtweets, 'r') as csvfile: 
        #reader = csv.reader(csvfile, delimiter='\t')
        for row in csvfile:
            row = row.strip("\n")
            columns = row.split("\t")
            author = columns[0]
            date = columns[1]
            party = columns[2]
            content = columns[3]
            if author == '"AUTEUR': 
                continue
            if date == 'DATUM':
                continue
            if party == 'PARTIJ': 
                continue
            if content == 'CONTENT"':
                continue
            contentlist.append(content)
            authorlist.append(author)
            partylist.append(party)
            datelist.append(date)
            

    return authorlist, datelist, partylist, contentlist
authorlist, datelist, partylist, contentlist = read_csv(newtweets) #date, party, content = read_csv(filename)

print('filtering non-dutch tweets...')
def get_only_dutch(author, date, party, content):
    """
    Using the langdetect package, it only gets the tweets written in dutch. 
    It also gets rid of tweets that have less than 25 characters. 
    """
    language = []
    for tweet in content:
        language.append(detect(tweet))
    
    zipped = list(zip(author, date, party, content, language))
    nested_list = [list(item) for item in zipped]

    only_dutch = []
    for item in nested_list:
        if 'nl' in item:
            only_dutch.append(item)
    
    only_longer_tweets = []
    for item in only_dutch: 
        if len(item[3]) > 25:
            only_longer_tweets.append(item)
            
    author = []
    date = []
    party = []
    content = []
    for item in only_longer_tweets:
        author.append(item[0])
        date.append(item[1])
        party.append(item[2])
        content.append(item[3])
        
    return author, date, party, content

author, date, party, content = get_only_dutch(authorlist, datelist, partylist, contentlist)

print('preprocessing tweets...')
def clean_tweets(content):
    """
    Using output of 'read_csv' or 'read_csv_2nd_way' function, cleans author list 
    to leave behind pure content only.
    Module to use: regex, 'import re'
    """
    cont=[]
    for item in content: 
        first = re.sub(r'^.*?:', '', item)
        cont.append(first)
    cleaned1=[]
    for item in cont:
        links = re.sub(r"http\S+",'', item)
        cleaned1.append(links) 
    cleaned2=[]
    for item in cleaned1:
        users = re.sub("@[^:]*:", '', item) #removes the users that posted/retweeted, but not the users mentioned 
                                            #inside the tweet
        cleaned2.append(users)
        #users = re.sub(r'@\S+','', item)  #this line removes all users, but we want to keep the ones in the tweet.
    check_nr_tweets = len(cleaned2)
   
    cleaned3 = []
    for item in cleaned2:
        unicodes=re.sub(r'(\\u[0-9A-Fa-f]+)','', item)
        #unicodes=item.replace('\u2066','').replace('\u2069','').replace('\xa0', '')
        cleaned3.append(unicodes)
    
    cleaned4 = []
    for item in cleaned2: 
        rt = re.sub(r'RT', '', item)
        cleaned4.append(rt)
    
    return cleaned4

cleaned = clean_tweets(content)
x_test = cleaned
#--------------------- OPEN TRAINING & TEST DATA--------------------------------------------
def read_csv(filename, sheetname='sheet'): 
    """
    Reads excel file containing training data and 
    assigns tweets and labels as x or y variable values.
    """
    df = pd.read_excel(open(filename, 'rb'),
               sheet_name=sheetname) 
    x = df.values[:,0]
    y = df.values[:,1]
  
    
    return x, y
x_train_binary, y_train_binary = read_csv('s/gold.xlsx', 'gold-proactivity')
#x_test_binary, y_test_binary = read_csv(inputpath1, sheetname1) #'gold.xlsx', 'test-data-proactivity')
x_train_sentiment, y_train_sentiment = read_csv('s/gold.xlsx', 'gold-polarity')
#x_test_sentiment, y_test_sentiment = read_csv(inputpath2, sheetname2) #'gold.xlsx', 'test-data-proactivity')


## ----------------------------- RUN MODEL  --------------------------------------------
def svm_classifier(Xtrain, testdata, Ytrain): 
    """
    Trains the SVM classifier and returns predicted labels on unseen tweets. 
    """
    vectorizer = TfidfVectorizer()
    x_train_vectors = vectorizer.fit_transform(Xtrain)  
    x_test_vectors = vectorizer.transform(testdata)
    
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train_vectors,Ytrain)
    
    predict = clf.predict(x_test_vectors)
    
    return predict


sentiment = svm_classifier(x_train_sentiment, x_test, y_train_sentiment )
proactivity = svm_classifier(x_train_binary, x_test, y_train_binary )

##--------------------------- CONCATENATE LABELS WITH TWEETS -----------------
def concatenate_for_json(tweets, labels1, labels2):
    """
    concatenates predictions of both systems with the original tweet
    returns a dictionary ready to be loaded as a json format
    """
    
    zipped = list(zip(labels1, labels2))
    nested_list = [list(item) for item in zipped]
    keys = tweets 
    values = nested_list
    dictionary = dict(zip(keys, values))
    
    return dictionary 
dictionary = concatenate_for_json(x_test, sentiment, proactivity)

print('saving output labels to json file')
## --------------------------- OUTPUT TO JSON FILE ------------------------
with open("system_output.json", "w") as outfile:
     json.dump(dictionary, outfile)
      

