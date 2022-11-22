# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:10:42 2022

@author: henry
"""


"""
Load datasets
"""

import pandas as pd
import re 
import numpy as np
from tqdm import tqdm
import nltk

from gensim.models import Word2Vec
from scipy import spatial #for cosine similarity


# remove HTML stuff
# https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
def remove_html_tags(text):
    clean = re.compile('<.*?>|\\\\n')
    return(re.sub(clean, '', text))


weblink = "https://raw.githubusercontent.com/casualcomputer/AgPal/master/agpal_search_result/agpal_search_results_run2.csv"
forum_posts = pd.read_csv(weblink).iloc[:,1:]
forum_posts.iloc[:,:7] = forum_posts.iloc[:,:7].applymap(lambda st: st[st.find("[")+1:st.find("]")])
forum_posts.description = forum_posts.description.apply(remove_html_tags)  
forum_posts.organization = forum_posts.organization.apply(remove_html_tags) 
forum_posts.contact = forum_posts.contact.apply(remove_html_tags) 
forum_posts.reset_index(drop=True)
forum_posts.head()

  
"""
Data pre-processing
"""
# Get stop words 
en_stop = set(nltk.corpus.stopwords.words('english'))

# Lemmatization
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

# Text cleaning function for gensim fastText word embeddings in python
def process_text(document):
     
        # Remove extra white space from text
        document = re.sub(r'\s+', ' ', document, flags=re.I)
         
        # Remove all the special characters from text
        document = re.sub(r'\W', ' ', str(document))
 
        # Remove all single characters from text
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
 
        # Converting to Lowercase
        document = document.lower()
 
        # Word tokenization       
        tokens = document.split()
        # Lemmatization using NLTK
        lemma_txt = [stemmer.lemmatize(word) for word in tokens]
        # Remove stop words
        lemma_no_stop_txt = [word for word in lemma_txt if word not in en_stop]
        # Drop words 
        tokens = [word for word in tokens if len(word) > 3]
                 
        clean_txt = ' '.join(lemma_no_stop_txt)
 
        return clean_txt
    
some_sent = forum_posts.description
clean_corpus = [process_text(sentence) for sentence in tqdm(some_sent) if sentence.strip() !='']
clean_corpus_lst = [d.split() for d in clean_corpus]
 
word_tokenizer = nltk.WordPunctTokenizer()
word_tokens = [word_tokenizer.tokenize(sent) for sent in tqdm(clean_corpus)]


"""
Train Word2Vec with fasttest: word embedding: https://thinkinfi.com/fasttext-word-embeddings-python-implementation/#comments
"""

'''
from gensim.models.fasttext import FastText

# Defining values for parameters
embedding_size = 300
window_size = 5
min_word = 5
down_sampling = 1e-2
 
fast_Text_model = FastText(word_tokens,
                      vector_size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      workers = 4,
                      sg=1,
                      epochs=100)


# Save fastText gensim model
fast_Text_model.save("model/agpal_word_embed_v0")

'''
# Load saved gensim fastText model
fast_Text_model = Word2Vec.load("model/agpal_word_embed_v0")
# The folder "model" contains 2 files:
# download "model/agpal_word_embed_v0" at https://storage.googleapis.com/fasttext_models/agpal_word_embed_v0
# download "model/agpal_word_embed_v0.wv.vectors_ngrams.npy" at https://storage.googleapis.com/fasttext_models/agpal_word_embed_v0.wv.vectors_ngrams.npy

"""
Measure similarity between sentences
#https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
"""


#http://yaronvazana.com/2018/09/20/average-word-vectors-generate-document-paragraph-sentence-embeddings/
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in list(word2vec_model.wv.index_to_key)]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        print("Can't map description to a word vector!")
        return []


# Ask user for search string and get word2Vec representation of search string
search_string = input("enter your search keywords: ") #text that goes into the search bar
search_string_as_list = search_string.split()
search_string_word_vec = get_mean_vector(fast_Text_model,search_string_as_list)


# Get word2Vec representation of listings 
word_vec_lst = []
for listing in clean_corpus_lst:
    listing_word_vec = get_mean_vector(fast_Text_model,listing)
    if np.shape(listing_word_vec)[0] ==300:
        cos_sim = 1 - spatial.distance.cosine(search_string_word_vec, listing_word_vec)
    if np.shape(listing_word_vec)[0] ==0:
        cos_sim = 0
    word_vec_lst.append(cos_sim)


#Assign similarity scores to the orginal dataset
forum_posts['cos_sim']= word_vec_lst

print("ranking of websties (most relevant to least relevant):")
print(forum_posts.sort_values(by='cos_sim',ascending=False))
forum_posts.sort_values(by='cos_sim',ascending=False).to_csv("keyword_search_results.csv")

#failed to map ['toxicity', 'walnut', 'discussed'] to a word vector
#can add tfid as well: https://www.machinelearningplus.com/nlp/cosine-similarity/ (to-do)

"""
TSNE data visualization
"""

'''
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# tsne plot for below word
# for_word = 'food'
def tsne_plot(for_word, w2v_model):
    # trained fastText model dimention
    dim_size = w2v_model.wv.vectors.shape[1]
 
    arrays = np.empty((0, dim_size), dtype='f')
    word_labels = [for_word]
    color_list  = ['red']
 
    # adds the vector of the query word
    arrays = np.append(arrays, w2v_model.wv.__getitem__([for_word]), axis=0)
 
    # gets list of most similar words
    sim_words = w2v_model.wv.most_similar(for_word, topn=10)
 
    # adds the vector for each of the closest words to the array
    for wrd_score in sim_words:
        wrd_vector = w2v_model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
 
    #---------------------- Apply PCA and tsne to reduce dimention --------------
 
    # fit 2d PCA model to the similar word vectors
    model_pca = PCA(n_components = 10).fit_transform(arrays)
 
    # Finds 2d coordinates t-SNE
    np.set_printoptions(suppress=True)
    Y = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(model_pca)
 
    # Sets everything up to plot
    df_plot = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words_name': word_labels,
                       'words_color': color_list})
 
    #------------------------- tsne plot Python -----------------------------------
 
    # plot dots with color and position
    plot_dot = sns.regplot(data=df_plot,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df_plot['words_color']
                                 }
                    )
 
    # Adds annotations with color one by one with a loop
    for line in range(0, df_plot.shape[0]):
         plot_dot.text(df_plot["x"][line],
                 df_plot['y'][line],
                 '  ' + df_plot["words_name"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df_plot['words_color'][line],
                 weight='normal'
                ).set_size(15)
 
 
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
 
    plt.title('t-SNE visualization for word "{}'.format(for_word.title()) +'"')
    
tsne_plot(for_word='corp', w2v_model=fast_Text_model)
'''

# interesting links: https://www.youtube.com/watch?v=fptTLo8JZDg
# similarity between doucments: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# https://medium.com/@hari4om/word-embedding-d816f643140