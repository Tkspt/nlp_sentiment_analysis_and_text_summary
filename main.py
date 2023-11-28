import streamlit as st
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from summarizer import resume_text, with_textRankSummarizer, with_lexRankSummarizer, with_lsaSummarizer

def load_tokenizer(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = file.read()
        tokenizer = tokenizer_from_json(data)
    return tokenizer

english_sentiment_analysis_model = load_model('./models/english_sentiment_analysis.h5', compile=False)
french_sentiment_analysis_model = load_model('./models/french_sentiment_analysis.h5', compile=False)

st.sidebar.header("Menu")

st.sidebar.subheader("Sentiment analysis")
sent_language = st.sidebar.selectbox(
    'Choose your language',
    ('choose language', 'English', 'French'),
    key = "sent",
)

st.sidebar.subheader("Text summarization")
sum_language = st.sidebar.selectbox(
    'Choose your language',
    ('choose language','English', 'French'),
    key = "sum",
)

if(sent_language == 'French') :
    st.title("Analyseur de Sentiment")
    user_text = st.text_input("Saisissez votre phrase ici")
    
    if(user_text != ""):
        tokenizer = load_tokenizer('french_tokenizer.json')
        sentence = pad_sequences(tokenizer.texts_to_sequences([user_text]), 25)
        
        y_pred= french_sentiment_analysis_model.predict(sentence)
        y_pred = np.round(y_pred)
        
        result = "Sentiment Negative" if int(y_pred[0])== 1 else "Sentiment Positive"
        
        st.write(result)

if(sent_language == 'English') :
    st.title("Sentiment analysis")
    user_text = st.text_input("Write your sentence here")
    
    if(user_text != ""):
        tokenizer = load_tokenizer('english_tokenizer.json')
        sentence = pad_sequences(tokenizer.texts_to_sequences([user_text]), 21)
        
        y_pred= english_sentiment_analysis_model.predict(sentence)
        y_pred = np.round(y_pred)
        
        result = "Negative Sentiment" if int(y_pred[0])== 1 else "Positive Sentiment"
        
        st.write(result)
    

if(sum_language == 'French') :
    st.title("Résumé de text français")
    user_text = st.text_area("Saisissez votre text ici")
    resume_text_btn = st.button("model fait maison")
    textRankSummarizer_btn = st.button("model textRankSummarizer")
    lexRankSummarizer_btn = st.button("model lexRankSummarizer")
    lsaSummarizer_btn = st.button("model lsaSummarizer")
    summary_size = 5
    summary = ""
    
    if(resume_text_btn):
        summary = resume_text(user_text, resume_size = summary_size, language = sum_language.lower())
    elif(textRankSummarizer_btn):
        summary = with_textRankSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    elif(lexRankSummarizer_btn):
        summary = with_lexRankSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    else:
        summary = with_lsaSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    
    st.write(summary)

if(sum_language == 'English') :
    st.title("English text summarizer")
    user_text = st.text_area("Write your text here")
    resume_text_btn = st.button("model from scratch")
    textRankSummarizer_btn = st.button("textRankSummarizer model")
    lexRankSummarizer_btn = st.button("lexRankSummarizer model")
    lsaSummarizer_btn = st.button("lsaSummarizer model")
    summary_size = 5
    summary = ""
    
    if(resume_text_btn):
        summary = resume_text(user_text, resume_size = summary_size, language = sum_language.lower())
    elif(textRankSummarizer_btn):
        summary = with_textRankSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    elif(lexRankSummarizer_btn):
        summary = with_lexRankSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    else:
        summary = with_lsaSummarizer(user_text, resume_size = summary_size, language = sum_language.lower())
    
    st.write(summary)