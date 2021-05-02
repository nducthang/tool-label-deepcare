import streamlit as st
import pandas as pd
from gensim import corpora
from gensim.summarization import bm25
import string
import pickle

# @st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('./data/data.csv')
    data['question_similaries'] = data['question_similaries'].apply(eval)
    return data

def preprocessing(row):
    row = row.translate(str.maketrans('', '', string.punctuation))
    return row.lower()

# @st.cache()
# def load_bm25():
#     with open('weight_bm25.pickle', 'rb') as handle:
#         params = pickle.load(handle)
#         return params

@st.cache(allow_output_mutation=True)
def load_bm25():
    data = load_data()
    lst_question = data['question'].apply(lambda x: preprocessing(str(x))).tolist()
    lst_answer = data['answer'].apply(lambda x: preprocessing(str(x))).tolist()
    lst_qa = lst_question + lst_answer

    texts = [item.split() for item in lst_qa]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)
    params = {"BM25": bm25_obj, "texts": texts, 'dictionary': dictionary}
    return params
