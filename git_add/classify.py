import csv
import re
from os import path
from os.path import exists
import pickle
from transformers import BertTokenizer, BertModel, BertConfig #need install transformers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #need install vaderSentiment
import os
import math
from bertembed import sentenceEmbedding, makeFeature, sentiment_scores

#reads the file that stores the features
#citation: https://thispointer.com/python-read-csv-into-a-list-of-lists-or-tuples-or-dictionaries-import-csv-to-list/
def readFileSimple(filepath_list):
    featureList_list = []
    for filepath in filepath_list:
        with open(filepath, 'r') as read:
            readerStore = csv.reader(read)
            featureList = list(readerStore)
            featureList_list.append(featureList)
    return featureList_list

#trains the model via a logistic regression
def logReg(featureList_list): #takes in list of BERT embeddings of suicidal and non suicidal
    #citation: code below is mostly from https://www.marktechpost.com/2021/02/12/logistic-regression-with-a-real-world-example-in-python/
    X = []
    y = []
    category_num = len(featureList_list)
    for num, featureList in enumerate(featureList_list):
        X = X + featureList
        for i in range(len(featureList)):
            y.append(num+1) #1 is passed, 2 is not passed
    classifier = None
    if category_num == 2:
        classifier = LogisticRegression(random_state = 0, max_iter=5000)
    else:
        classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state = 0, max_iter=5000)
    classifier.fit(X, y)

    #save model below into a .sav file
    #citation: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    filename = 'classify_classifier.sav'
    pickle.dump(classifier, open(filename, 'wb'))

    # #using hold out data, predict suicisuicidetestde or non-suicide
    # y_pred = classifier.predict(X_test)
    #
    # #confusion tables
    # confusion = confusion_matrix(y_test, y_pred)
    # print('Confusion:',confusion)

#this function below is for use in classify
def bertFromLine(line):
    tensors = sentenceEmbedding(line)
    features = []
    for t in tensors:
        features.append(float(t))
    if len(features)!=768:
        print('bad embedding')
        return
    #x = sentiment_scores(clean(i))
    x = sentiment_scores(line)
    y = [float(x['neg']), float(x['neu']), float(x['pos'])]
    features = features+y
    return features

def classify_predictforline(line, category_num_index, filewriter, classifier, flag): #categorynum is indexed at 0 so category 1 is 0
    features = bertFromLine(line)
    if len(features)!=771:
        print('ERROR: Bad embedding, len not 771:', len(features))
        return -1
    else:
        features = [features]
        pred = classifier.predict_proba(features)
        predlist = (pred[0]).tolist()
        category_weight = predlist[category_num_index]

    #write line to file
    filewriter.write(str(category_weight) + " " + line + "\n")

    #return probability
    if flag == "":
        return category_weight
    elif flag == "v":
        return predlist
    return -1

def classify(filepath, category_num=1, outfile_path="classify_highprobabilitylines.txt", flag=""):
    category_num_index = category_num - 1
    classifier = pickle.load(open('classify_classifier.sav', 'rb'))

    #remove file if exists
    if exists(outfile_path):
        os.remove(outfile_path)

    #open file to append
    filewriter = open(outfile_path, "a")

    with open(filepath) as f:
        num_valid_lines = 0
        probability_sum = 0
        probability_sum_list = [] #used only when flag == "v"
        f_info = f.read().split("\n")
        for line in f_info:
            if line == "\n" or line == " ":
                print("oh no")
                continue
            line_length = len(line)
            if line_length>512:
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
                    probability = classify_predictforline(line, category_num_index, filewriter, classifier, flag)
                    if not probability == -1:
                        num_valid_lines = num_valid_lines + 1
                        if flag == "":
                            probability_sum = probability_sum + probability

                        if flag == "v":
                            if probability_sum_list == []:
                                probability_sum_list = probability
                            else:
                                for index, prob in enumerate(probability):
                                    probability_sum_list[index] = probability_sum_list[index] + prob
            else:
                probability = classify_predictforline(line, category_num_index, filewriter, classifier, flag)
                if not probability == -1:
                    num_valid_lines = num_valid_lines + 1
                    if flag == "":
                        probability_sum = probability_sum + probability

                    if flag == "v":
                        if probability_sum_list == []:
                            probability_sum_list = probability
                        else:
                            for index, prob in enumerate(probability):
                                probability_sum_list[index] = probability_sum_list[index] + prob
    if flag == "":
        print(f"This file is {probability_sum/num_valid_lines}% category {category_num}")
    elif flag == "v":
        for index, prob in enumerate(probability_sum_list):
            print(f"This file is {probability_sum_list[index]/num_valid_lines}% category {index+1}")
    filewriter.close()

basepath = os.getcwd()
numdirs = 2

#run logistic regression tests for evaluation, does not have to run right after bertDicSimpler
embeddingspaths = []
for i in range(numdirs):
    embeddingspaths.append("category"+str(i+1)+"embeddings.txt")
featureList_list = readFileSimple(embeddingspaths)
logReg(featureList_list)

#Try to classify
filepath = path.join(basepath, "category1", r"Copy of #1.Ledford.txt")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = BertModel.from_pretrained('bert-base-uncased', config=config)
category_num = 2 #which category do you want to look at?
#optional if set flag to "v" then print out all categories
classify(filepath, category_num, flag="v")
