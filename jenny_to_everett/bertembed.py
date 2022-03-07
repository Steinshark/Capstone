import csv
import re
from transformers import BertTokenizer, BertModel, BertConfig #need install transformers
from os import path
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #need install vaderSentiment
import os
import math
from cleantextstring import *

#returns bert embeddings of a sentence
def sentenceEmbedding(sentence):
    inputs = tokenizer(sentence, return_tensors = 'pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return list(last_hidden_states[0][0])

#param=[[home/file1,home/file2],[home/file3,home/file4]] where it has category 1 and 2
#these have list of full filepaths
def dataExtract(filepathlist_list):
    dataDict = {}
    for i, filepathlist in enumerate(filepathlist_list):
        dataDict["category"+str(i+1)] = []
        for filepath in filepathlist:
            with open(filepath) as f:
                f_info = f.read().split("\n")
                f_info = list(filter(None, f_info))
                #<add processing here>
                for line in f_info:
                    dataDict["category"+str(i+1)].append(cleantextstring(line))
    return dataDict

def makeFeature(line):
    line = re.sub(r"\#|\?|\*|\\n|\,|\.", "", line)
    tensors = sentenceEmbedding(line) #embed the paragraph

    #Make features for line
    features = []
    for t in tensors:
        features.append(float(t))
    if len(features)!=768:
        print('bad embedding')
        return -1
    x = sentiment_scores(cleantextstring(line)) #add vader sentiments
    y = [float(x['neg']), float(x['neu']), float(x['pos'])]
    features = features+y

    if len(features) == 771:
        return features
    else:
        print("vader error")
        return -1

#takes the dictionary of our data and then vectorizes them using a transformer
def bertFromDict(category_line_list):
    featuresList = []
    for line in category_line_list:
        line_length = len(line)
        if line_length>512: #in case the sentence is too long
            list_len = math.ceil(line_length / 512.0)
            lines = []
            for i in range(list_len):
                shortline = line[:512]
                if len(line) > 512: #long section still
                    if not line[512] == " ":
                        shortline = shortline.rsplit(' ', 1)[0]
                last_char_index = len(shortline)
                line = line[last_char_index+1:]
                lines.append(shortline[:last_char_index])
            for line in lines:
                features = makeFeature(line)
                if not features == -1:
                    featuresList.append(features)
        else:
            features = makeFeature(line)
            if not features == -1:
                featuresList.append(features)

    #store results of the embeddings into a file corresponding to the category
    filename = category +"embeddings.txt"
    with open(filename, "w") as f:
        wr = csv.writer(f)
        wr.writerows(featuresList)

#this function takes a sentence and then performs a sentiment analysis using a twitter vader sentiment tool
def sentiment_scores(sentence):
    #create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    #the SentimentIntensityAnalyzer object method gives a sentiment dictionary
    #which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict

if __name__ == '__main__':
    basepath = os.getcwd()
    numdirs = 2
    #Find names of dir
    filepathlist_list = []
    for i in range(numdirs):
        dirpath = basepath+"/category"+str(i+1)
        filepathlist = []
        for file in os.listdir(dirpath):
            filepath = path.join(dirpath, file)
            filepathlist.append(filepath)
        filepathlist_list.append(filepathlist)

    #Extract data
    dataDict = dataExtract(filepathlist_list)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = BertModel.from_pretrained('bert-base-uncased', config=config)

    #Get features of suicide data #processing
    for i in range(numdirs):
        category = "category"+str(i+1)
        bertFromDict(dataDict[category])
