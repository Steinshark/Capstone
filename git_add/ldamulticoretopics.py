import matplotlib.pyplot as plt   # we had this one before
import re
import gensim
import gensim.corpora as corpora
from os.path import exists
import os
from cleantextstring import *

#Uses LDA Multi-Core Topic Modeling to print out specified topics and a specific number of words from that topic to a specifiable file
def ldamulticoretopics(filepath_list, num_topics=10, num_words=3, outfile_path="ldamulticoretopics_out.txt"):
    whole_text = ""
    try:
        for filepath in filepath_list:
            with open(filepath) as f:
                whole_text = whole_text + " " + f.read()
    except:
        print("ERROR in ldamulticoretopics: Could not open file", filepath)
        exit(2)
    #split by end punctuation into sentences
    sentences_list = re.split('[\.\?\!\\n]\s*', whole_text)

    #create list of word lists
    data_words = []
    for sentence in sentences_list:
        sentence = cleantextstring(sentence)
        if sentence != "":
            data_words.append([sentence])

    # Create dictionary
    data_dict = corpora.Dictionary(data_words)

    # Create corpus
    texts = data_words

    # Term Document Frequency
    corpus = [data_dict.doc2bow(text) for text in texts]

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=data_dict,
                                           num_topics=num_topics)

    topics = lda_model.show_topics(num_topics, num_words)
    if exists(outfile_path):
        os.remove(outfile_path)
    for tuple in topics:
        with open(outfile_path, "a") as w:
            w.write(f"Category {str(tuple[0])}: {tuple[1]}\n\n")

filepath_list = ["category1/Copy of #1.Ledford.txt", "category1/Copy of #3.Ledford.txt"]
num_topics = 10
ldamulticoretopics(filepath_list, num_topics, 3, "ldamulticoretopics_out.txt")
