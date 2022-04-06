import streamlit as st
import numpy as np
#from tensorflow.keras.models import load_model
#import numpy as np
import pandas as pd
#import re
#import nltk
#from nltk.corpus import stopwords
#import seaborn as sns
import pickle
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Sentiment Analyzer',layout='centered', initial_sidebar_state='expanded')


#def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    #with open(file_name) as f:
        #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    #local_css("css/styles.css")
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities2 = ["Homepage","Analyzer"]
    st.sidebar.markdown("# Model?")
    choice2 = st.sidebar.selectbox("Choose among the given options:", activities2)

    if choice2=='Analyzer':
        st.markdown('<h2 style="color:black;" align="center">Sentiment Analysis of Hindi Movie Reviews</h2>', unsafe_allow_html=True)
        activities = ["Random Forest Classification","Decision Tree Classifier"]
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.markdown("#### Model?")
        choice = st.selectbox("Choose among the given options:", activities)
        
        if choice == 'Random Forest Classification':
            st.markdown("#### Enter the hindi movie review  here⬇")
            text = st.text_area("Please do not leave the text area empty")
        
            if text is not None:
                if st.button('Process'):
                    stop = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']
                    punctuations = ['nn','n', '।','/', '`', '+', '?', '$', '@', '[', '_', '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', "#", '*', '', ';', '-', '}','|']
                    to_be_removed = stop + punctuations
                    text = text.lower().replace('।'," ")
                    removed_stopword = []
                    for word in str(text).split(): 
                        if word not in to_be_removed:
                            removed_stopword.append(word)
                    text = np.array([" ".join(removed_stopword)])
                    st.write("After stopwords removal: ",text[0])
 
                    #tfidfconverter = TfidfVectorizer(max_features=200, min_df=1, max_df=0.10) 
                    tfidfconverter=pickle.load(open("models/tfidf.pkl", 'rb')) 
                    x = tfidfconverter.transform(text).toarray()
                    #models = ['rfc.pkl']
                    
                    model=pickle.load(open('models/rfc.pkl','rb'))
                    #st.markdown(i)
                    y=model.predict(x)
                    if y=='negative':
                        st.write('Predicted sentiment: Negative')
                    else:
                        st.write('Predicted sentiment: Positive')
                        
        if choice == 'Decision Tree Classifier':
            st.markdown("### Enter the hindi movie review  here⬇")
            text = st.text_area("Please do not leave the text area empty")
        
            if text is not None:
                if st.button('Process'):
                    stop = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']
                    punctuations = ['nn','n', '।','/', '`', '+', '?', '$', '@', '[', '_', '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', "#", '*', '', ';', '-', '}','|']
                    to_be_removed = stop + punctuations
                    text = text.lower().replace('।'," ")
                    removed_stopword = []
                    for word in str(text).split(): 
                        if word not in to_be_removed:
                            removed_stopword.append(word)
                    text = np.array([" ".join(removed_stopword)])
                    st.write("After stopwords removal: ",text[0])
 
                    #tfidfconverter = TfidfVectorizer(max_features=200, min_df=1, max_df=0.10) 
                    tfidfconverter=pickle.load(open("models/tfidf.pkl", 'rb')) 
                    x = tfidfconverter.transform(text).toarray()
                    model=pickle.load(open('models/dt.pkl','rb'))
                    #st.markdown(i)
                    y=model.predict(x)
                    if y=='negative':
                        st.write('Predicted sentiment: Negative')
                    else:
                        st.write('Predicted sentiment: Positive')
    if choice2=='Homepage':
        #st.markdown('<h2 style="color:white;" align="center">Sentiment Analysis of Hindi Movie Reviews</h2>', unsafe_allow_html=True)
        #st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHcPmimTJL2f4KBVmtG3QPYjHgVVGDoywevA&usqp=CAU
        st.markdown(
                """
                <style>
                .reportview-container {
                    background: url("https://miro.medium.com/max/1400/1*PFI22lMXZFyPpM3wm-IzeQ.jpeg");
                    background-size: 120% 100%;
                    background-repeat: no-repeat;
                    
                    }
                .sidebar{
                background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvkBtWCfjJ4xVqGIltIy8koeOoMZeU4d4dcQ&usqp=CAU");
                }
                </style>
                """,
                unsafe_allow_html=True
                    )   
    


main()
