#https://datascience.blog.wzb.eu/2016/06/17/creating-a-sparse-document-term-matrix-for-topic-modeling-via-lda/
import re
import nltk
from nltk.corpus import stopwords
from itertools import product
from string import ascii_lowercase
import numpy as np
from scipy.sparse import coo_matrix
import lda # had to pip install
import os
from os import path
from os.path import exists
from cleantextstring import *

def sparseldamatrix_topics(filepath_list, num_topics):
    #Create a dicitonary that maps the document name to the list of important words
    file_dict = {}
    for filepath in filepath_list:
        with open(filepath) as f:
            list_of_sentences = f.readlines()
            new_list_of_sentences = []
            for sentence in list_of_sentences:
                if sentence == "\n":
                    continue
                new_sentence = cleantextstring(sentence)
                new_sentence = re.sub("\n", " ", new_sentence)
                if new_sentence != "" and new_sentence != " ":
                    new_list_of_sentences.append(new_sentence)
            file_dict[file] = new_list_of_sentences

    # Convert data in dictionary of key: filenames, values: list of words
    filenames = list(file_dict.keys()) # Make array of file names to be the rows of the DTM
    filenames = np.array(filenames) # Convert to NumPy array

    # Make set of unique vocab over all the interviews
    num_unique_vocab = 0 #to help create matrix
    vocab = set()
    for words in file_dict.values():
        unique_words = set(words)    # all unique terms of this file
        vocab |= unique_words           # set union: add unique terms of this file
        num_unique_vocab += len(unique_words)  # add count of unique terms in this file
    vocab = np.array(list(vocab)) # Convert vocab to NumPy array

    # Create array to hold indices that sort vocab and help count the terms
    vocab_sorter = np.argsort(vocab)

    # Dimensions of DTM
    num_files = len(filenames)
    num_vocab = len(vocab)

    '''
    Create three arrays for the scipy.sparse.coo_matrix (COOrdinate format), which
    takes 3 parameters: data (entries), row indices of matrix, col indices of matrix

    Specify the data type to C integer (32 bit), which is less than the default np
    64 bit float, which takes a lot of memory
    '''
    data = np.empty(num_unique_vocab, dtype=np.intc)
    rows = np.empty(num_unique_vocab, dtype=np.intc)
    cols = np.empty(num_unique_vocab, dtype=np.intc)

    # Loop through the documents to fill the 3 above arrays for the matrix
    index = 0 # current index in the sparse matrix data
    for filename, word_list in file_dict.items():
        insert_points = np.searchsorted(vocab, word_list, sorter=vocab_sorter)
        word_indices = vocab_sorter[insert_points]

        # Count the unique terms of the document and get their vocab indices
        unique_index_list, unique_word_count_list = np.unique(word_indices,return_counts=True)
        num_unique_indices = len(unique_index_list)
        index_fill_end = index + num_unique_indices

        data[index:index_fill_end] = unique_word_count_list
        cols[index:index_fill_end] = unique_index_list
        index_file = np.where(filenames == filename) #index_file is the index of the particular file
        rows[index:index_fill_end] = np.repeat(index_file, num_unique_indices)

        index = index_fill_end # counter continues into next file to add more

    # Create the coo_matrix
    dtm = coo_matrix((data, (rows,cols)), shape=(num_files, num_vocab), dtype=np.intc)

    model = lda.LDA(n_topics=num_topics, n_iter=1000, random_state=1)
    model.fit(dtm)
    topic_word = model.topic_word_

    if exists("sparseldamatrix_topics.txt"):
        os.remove("sparseldamatrix_topics.txt")

    with open ("sparseldamatrix_topics.txt", "w") as writer:
        for i, topic_dist in enumerate(topic_word):
            writer.write(f"Topic {i}:\n")
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_topics+1):-1]
            writer.write(str(topic_words) + "\n")

#Read in text from category 1 directory to run code on
folder = "category1"
basepath = os.getcwd()
dirpath = path.join(basepath, folder)

#list of absolute paths
filepath_list =[]
for file in os.listdir(dirpath):
    filepath = path.join(dirpath, file)
    filepath_list.append(filepath)
num_topics = 3 #set by user on how many categories of topics they want
sparseldamatrix_topics(filepath_list, num_topics)
