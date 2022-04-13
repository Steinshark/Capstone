#!/usr/bin/env python
# coding: utf-8

# Document Clustering
# Author: Krystal Kim
from scipy.sparse import lil_matrix  # , save_npz, load_npz
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import re

class DocClusterer:
    def __init__(self):
        '''
        try:
            # doc init
            self.f_docword = open("docword.interviews.txt", "r")
            self.lines = self.f_docword.readlines()
            # vocab init
            self.vocab_file = open("vocab.interviews.txt", "r")
            self.vocab_dict = {}
            i = 1  # line number
            for word in self.vocab_file:
                word = word.strip()
                self.vocab_dict[i] = word
                i += 1
            # print(vocab_dict)

            # hardcoded for now
            self.num_docs = 22
            self.num_words = 3321
            self.k = 2

        except:
            self.f_docword = None
            print("Req'd files do not exist.")
        '''

        self.n_clusters = 2 
        self.batch_size = 32

    # We then initialize the lil sparse matrix (docID x wordID) and fill them with the word counts.
    # The rows represent the documents (docID) and the columns represent individual words (wordID).
    # Afterwards, we convert the lil matrix to a csr to easily access the rows (documents).
    def run(self, APP_REF):
        # testing version (1st trail of removing stopwords)
        # self.my_lil = lil_matrix((22+1, 3321+1))
        self.my_lil = lil_matrix(
            (self.num_docs+1, self.num_words+1))  # general version

        # populate the lil matrix
        for line in self.lines:
            docID, wordID, count = line.split()
            self.my_lil[int(docID), int(wordID)] = int(count)

        self.my_csr = self.my_lil.tocsr()
        # make K means model - default value should be 2
        self.kmeansModel = MiniBatchKMeans(n_clusters=self.n_clusters,
                                           random_state=0,
                                           self.batch_size=32)

        self.kmeansModel.fit(self.my_lil)
        self.clusters = self.kmeansModel.predict(self.my_lil)
        self.cluster_centers = self.kmeansModel.cluster_centers_

        # Using the cluster_centers, in each row (quintessential document
        # of a particular cluster), I wanted to find the 10 largest counts'
        # indices (wordID) so that I can map them to actual words. I
        # thought this would give me the top 10 most commonly used words
        # in that particular cluster:

        ix = 1
        for doc in self.cluster_centers:
            # grab top 10 word IDs
            top10_wid = sorted(range(len(doc)), key=lambda sub: doc[sub])[-10:]

            # map wid to actual words
            comn_word_list = []
            for wid in top10_wid:
                comn_word_list.append(self.vocab_dict[wid])

            # **** PRINTING THE TOP 10 WORDS IN THE CLUSTER ****
            print(f"Cluster {ix}: {comn_word_list}")
            ix += 1

        print(f'Number of cluster documents: {self.k}')


    def run_new(self,APP_REF):
        
        # Create a vocab file, docwords, 
        # Find number of docs, number of words 
        docwords,vocab,n_docs,n_words = create_vocab(APP_REF.data['loaded_files'])

        print(f"found vocab of size {len(vocab)}\nwith {len(n_docs)} documents and {len(n_words)} words")

        # Create and fill a matrix to hold the docwords 
        dword_matrix = lil_matrix((n_docs+1,n_words+1))
        for line in docwords:
            docID, wordID, count = line.split()
            dword_matrix[int(docID), int(wordID)] = int(count)

        # Convert to csr matrix 
        self.dword_matrix = dword_matrix.tocsr()

        # make K means model - default value should be 2
        self.kmeansModel = MiniBatchKMeans(n_clusters=self.n_clusters,
                                           random_state=0,
                                           self.batch_size=32)

        # Run the model on the data
        self.kmeansModel.fit(self.dword_matrix)
        self.clusters = self.kmeansModel.predict(self.dword_matrix)
        self.cluster_centers = self.kmeansModel.cluster_centers_


        # Show info from the model
        for i, doc in enumerate(self.cluster_centers):

            # grab top 10 word IDs
            top10_wid = sorted(range(len(doc)), key=lambda sub: doc[sub])[-10:]

            # map wid to actual words
            comn_word_list = []
            for wid in top10_wid:
                comn_word_list.append(vocab[wid])

            # **** PRINTING THE TOP 10 WORDS IN THE CLUSTER ****
            print(f"Cluster {i}: {comn_word_list}")

        print(f'Number of cluster documents: {self.n_clusters}')




def create_vocab(filepaths):
    punctuation = ".,-!?()#:0123456789"
    vocab_file = ''
    unique_words = {}

    docwords = ''
    files = {}

    # Find vocab
    for filename in filepaths:

        # Ensure filename is a string 
        assert isinstance(filename,str)

        # Find all unique words 
        try:
            symb_cleaned = re.sub(r"[^\w | \s | \n]", "", open(filename,'r').read())
            file_contents = re.split(r"\s+|\n",symb_cleaned)
            for word in file_contents:
                word = word.lower()
                if not word in unique_words:
                    unique_words[word] = len(unique_words)
                    vocab_file += f"{word}\n"
        except ValueError:
            print("oops")

    # Find word counts 
    for doc_num,filename in enumerate(filepaths,1):
        symb_cleaned = re.sub(r"[^\w | \s | \n]", "", open(filename,'r').read())
        file_contents = re.split(r"\s+|\n",symb_cleaned)

        for word in unique_words:
            if word in file_contents:

                wc = file_contents.count(word)
                wn = unique_words[word]

                docwords += f"{doc_num} {wn} {wc}\n"

    return docwords,unique_words,doc_num,len(unique_words)



# --- Solely for testing ---
if __name__ == '__main__':
    #kkimDC = DocClusterer()
    #kkimDC.run(1)
    create_vocab(["a.txt","b.txt","c.txt"])
