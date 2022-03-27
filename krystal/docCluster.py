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


class DocClusterer:
    def __init__(self):
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
        self.kmeansModel = MiniBatchKMeans(n_clusters=self.k,
                                           random_state=0,
                                           batch_size=32)

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


# --- Solely for testing ---
if __name__ == '__main__':
    kkimDC = DocClusterer()
    kkimDC.run(1)
