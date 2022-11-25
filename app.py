import streamlit as st
import pandas as pd
import nltk
from annotated_text import annotated_text
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from nltk.chunk import conlltags2tree, tree2conlltags
import spacy_streamlit
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

qa = pipeline('question-answering', model=model_name, tokenizer=model_name)

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

nlp = spacy.load("en_core_web_sm")

models = ["en_core_web_sm", "en_core_web_md"]
#nltk.download('punkt')

def QAFunction(text,question):
    QA_input = {
    'question': question,
    'context': text
    }
    res = qa(QA_input)

    return res


def distrillBert(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    return output

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def sentenceSegment(text):
    sentences = sent_tokenize(text)

    return sentences

def wordSegment(text):
    tokens = word_tokenize(text)

    return tokens

def PosTag(text):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    return tagged

def wordSegment2(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    return tokens

def NerTag(text):
    doc = nlp(text)
    ner = []
    for token in doc:
        ner.append({'text': token.text, 'IOB': token.ent_iob_, 'ent_type': token.ent_type_})
  

    return ner

def textClassification(text):
    tokens = word_tokenize(text)
    s = []
    for t in tokens:
        s.append(sentiment_analysis(t))

    return s

st.title('All-in-one NLP Webapp')

st.header('Text/Story/Passage')


txt = st.text_area('Your Text', ''' ''')


if st.button('save',key=1):
    st.write(txt)




st.header('Question')
question = st.text_input('Your question', '')
if st.button('ask',key=2):
    ans = QAFunction(txt,question)
    st.write(ans)



st.header('Summarize')
if st.button('summarize',key=3):
    summarize = summarizer(txt, max_length=130, min_length=30, do_sample=False)
    
    st.write(summarize)


st.header('Word segmentation')
option = st.selectbox(
'Select one',
('Text', 'Question'))

st.write('You selected:', option)
if st.button('word',key=4):
    if option == 'Text':
        words = wordSegment(txt)
        words2 = wordSegment2(txt)
        st.write(words)
        st.write(words2)
    elif option == 'Question':
        words = wordSegment(question)
        st.write(words)

st.header('Sentence segmentation')
option_1 = st.selectbox(
'Select one',
('Text', 'Question'),key=2)

st.write('You selected:', option_1)
if st.button('sentence',key=5):
    sentences = sentenceSegment(txt)
    st.write(sentences)


st.header('Sentence classification')
if st.button('sentence',key=20):
    sentences = textClassification(txt)
    st.write(sentences)


st.header('Sentence tokenizer')
option_2 = st.selectbox(
'Select one',
('Text', 'Question'),key=10)

st.write('You selected:', option_2)


if st.button('output',key=11):
    out = distrillBert(txt)
    st.write(out)


st.header('NER tagging')
if st.button('NER',key=6):
    ner = NerTag(txt)
    st.write(ner)

st.header('POS tagging')
option_2 = st.selectbox(
'Select one',
('Text', 'Question'),key=3)

if option == 'Text':
    pos = PosTag(txt)
elif option == 'Question':
    pos = PosTag(question)

if st.button('POS',key=7):
    st.write(pos)
    



if st.button('visualize POS',key=8):
    if option == 'Text':
        visualizers = ["ner", "textcat"]
        spacy_streamlit.visualize(models, txt, visualizers)
#     elif option == 'Question':
#         pos = PosTag(question)
#         annotated_text(pos)


# if st.button('draw POS',key=9):
#     if option == 'Text':
#         pos = PosTag(txt)
#         for p in pos:
#             annotated_text(p)