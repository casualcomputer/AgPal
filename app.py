import streamlit as st
from gensim.models import Word2Vec
import numpy as np 
import pickle
from scipy import spatial #for cosine similarity
import re
import pandas as pd

# remove HTML stuff
# https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
def remove_html_tags(text):
    clean = re.compile('<.*?>|\\\\n')
    return(re.sub(clean, '', text))


def get_AgPal_data():
    weblink = "https://raw.githubusercontent.com/casualcomputer/AgPal/master/agpal_search_result/agpal_search_results_run2.csv"
    dat = pd.read_csv(weblink).iloc[:,1:]
    dat.iloc[:,:7] = dat.iloc[:,:7].applymap(lambda st: st[st.find("[")+1:st.find("]")])
    dat.description = dat.description.apply(remove_html_tags)  
    dat.organization = dat.organization.apply(remove_html_tags) 
    dat.contact = dat.contact.apply(remove_html_tags) 
    dat.reset_index(drop=True)
    return dat
 

# Load saved gensim fastText model
@st.cache
def load_fasttext_model():
    return Word2Vec.load("model/agpal_word_embed_v0")
    

# Load AgPal listing
@st.cache
def load_AgPal_listing():
    # load AgPal listing (from list) 
    with open("AgPal_listings", "rb") as fp:   # Unpickling
       AgPal_listing = pickle.load(fp)
    return AgPal_listing


fast_Text_model = load_fasttext_model()
forum_posts = get_AgPal_data()
clean_corpus_lst = load_AgPal_listing()

# load helper functions
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words_lst = [word for word in words if word in list(word2vec_model.wv.index_to_key)]
    if len(words_lst) >= 1:
        return np.mean(word2vec_model.wv[words_lst], axis=0)
    else:
        return word2vec_model.wv[words][0]
        #[] #   #bug: sometimes return empty

# Load AgPal listing
@st.cache
def init_word_embedding():
    return [get_mean_vector(fast_Text_model,listing) for listing in clean_corpus_lst]

listing_word_vec = init_word_embedding()


# title and description
st.write("""
# AgPal+ dummy

""")

# search bar (ask for user input)
search_string = st.text_input("Search!", "agri food")
search_string_as_list = search_string.split()
search_string_word_vec = get_mean_vector(fast_Text_model,search_string_as_list)


#calculate similarity between user input and AgPal listing
#st.text(search_string_word_vec) #for debugging
forum_posts['cos_sim']= [1 - spatial.distance.cosine(search_string_word_vec, word_vec) if np.shape(word_vec)[0] ==300 else 0 for word_vec in listing_word_vec ]
top_five_results = forum_posts.sort_values(by='cos_sim',ascending=False)[["programTitle","description"]].head(10)
st.table(top_five_results)