#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 23:16:24 2021

@author: abhinav
"""

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string




app = Flask(__name__,template_folder='template')

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('/index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method == 'POST':
        
        tweet = request.form["tweet"]
        
        tweet = tweet.lower()
        
        word_collection = nltk.word_tokenize(tweet)
        
        lemmatizer = WordNetLemmatizer()
        word_collection = [lemmatizer.lemmatize(word) for word in word_collection if word not in set(stopwords.words("english"))]
        
        tweet = " ".join(word_collection)
        
        punctuations = set(string.punctuation)
        
        cleaned_tweet = ["".join(x for x in tweet if x not in punctuations)]
        
        
        final_tweet = vectorizer.transform(cleaned_tweet)
        
        prediction = model.predict(final_tweet)
        
        if(prediction == 1):
            return render_template('/index.html',prediction_text = "This tweet is classified as racist")
        else:
            return render_template('/index.html',prediction_text = "This tweet is not classified as racist")
        


if __name__ == '__main__':
    import warnings
    warnings.warn("use 'python -m app', not 'python -m app.app'", DeprecationWarning)
    main()
    app.run(debug=True)

        
        
        