import add_similary
import statistical
import streamlit as st
import add_sample
import BM25
import pandas as pd
from gensim import corpora
from gensim.summarization import bm25
import string


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


PAGES = {
    "Gán nhãn câu hỏi tương đồng": add_similary,
    "Thêm dữ liệu": add_sample,
    "Thống kê": statistical,
    "Search engine BM25": BM25
}


def preprocessing(row):
    row = row.translate(str.maketrans('', '', string.punctuation))
    return row.lower()


@st.cache(allow_output_mutation=True)
def init():
    data = pd.read_csv('./data/data.csv')
    data['question_similaries'] = data['question_similaries'].apply(eval)
    lst_question = data['question'].apply(
        lambda x: preprocessing(str(x))).tolist()
    lst_answer = data['answer'].apply(lambda x: preprocessing(str(x))).tolist()
    lst_qa = lst_question + lst_answer

    texts = [item.split() for item in lst_qa]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    bm25_obj = bm25.BM25(corpus)
    return data, texts, dictionary,   bm25_obj


data, texts, dictionary,  bm25_obj = init()
params = {"data": data, "BM25": bm25_obj,
          "texts": texts, 'dictionary': dictionary}

st.sidebar.title('PAGES')
selection = st.sidebar.selectbox('', list(PAGES.keys()))
page = PAGES[selection]
page.app(params)
