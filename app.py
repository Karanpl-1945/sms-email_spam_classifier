import streamlit as st
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
import nltk
from nltk.corpus import stopwords


# Download stopwords if not already present
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)

import os

base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')


tfidf = pickle.load(open(vectorizer_path, 'rb'))


model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms= st.text_input("Enter the message")

if st.button("predict"):

    #1. preprocessing

    transformed_sms=transform_text(input_sms)
    #2. vectorise
    vector_input=tfidf.transform([transformed_sms])

    #3. predict
    result=model.predict(vector_input)[0]
    #4.display
    if  result==1:
        st.header("spam")
    else:
        st.header("not spam")


