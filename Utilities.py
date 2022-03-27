# This class is meant to make the GUI_APP class cleaner
# "Utilities" will not be instantiated. All methods are
# static
import os                                           # allows access to filepath
import tkinter
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
import json

from scipy.sparse import lil_matrix  # , save_npz, load_npz
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf

import model, sample, encoder
import pandas as pd
import numpy as np
# NEEDED FOR MODELS 
import csv
import re
import pickle
from transformers import BertTokenizer, BertModel, BertConfig #need install transformers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #need install vaderSentiment
import math
import nltk
from nltk.corpus import stopwords
from itertools import product
from string import ascii_lowercase
import numpy as np
from scipy.sparse import coo_matrix
import lda # had to pip install
import matplotlib.pyplot as plt   # we had this one before
import re
import gensim
import gensim.corpora as corpora
import spacy

nlp  = spacy.load('en_core_web_sm')
# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb,txt=None):
        self.filepath = filepath
        if not txt is None:
            self.contents_as_rb = None
            self.contents_as_text = txt
        else:
            self.contents_as_rb = contents_as_rb
            self.contents_as_text = contents_as_rb.decode()

        self.lines = [l for l in self.contents_as_text.split('\n') if not l == '' and not l == '\n']
        self.words = [w for w in self.contents_as_text.split(' ')  if not w == '']
        self.chars = [c for c in self.contents_as_text  if not c == ' ' or not c =='']


    #def __dict__(self):
    #    return {'fp' : self.filepath,'rb' : self.contents_as_rb,'txt':self.contents_as_text,
    #            'lines':self.lines,'words':self.words,'chars' : self.chars}
    def __repr__(self):
        return 'new_imported_file\n\t' + str(self.filepath) + '\n\t' + str(self.contents_as_rb) + '\n'


class Utilities:
    supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )
    @staticmethod
    def get_os_root_filepath():
        return os.getcwd()


    @staticmethod
    def get_window_size_as_text(APP_REFERENCE):
        text = str(APP_REFERENCE.settings['init_width'])
        text += 'x'
        text += str(APP_REFERENCE.settings['init_height'])
        return text


    # Upload multiple files to app
    @staticmethod
    def import_files(APP_REFERENCE):

        supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )

        # Opens blocking tkinter file dialog
        file_list = askopenfiles(mode='rb',filetypes=supported_types)


        # user picks a file which is added to the data dictionary of the APP
        distinct_files = 0

        if not len(file_list) == 0:
            for file in file_list:
                if not file is None:
                    distinct_files += 1
                    # Add to the running instance's data  dictionary
                    APP_REFERENCE.data['loaded_files'].append(ImportedFile(file.name,file.raw.read()))
        Utilities.save_session(APP_REFERENCE)

    # Upload single file to app
    @staticmethod
    def import_file(APP_REFERENCE,work_block=None):
        supported_types =   (   ("text files", "*.txt"),\
                                ("word files", "*.docx"),\
                                ("pdf files", "*.pdf"),\
                                # Probably will not include in final version
                                ("all files", "*.*")  )

        file = askopenfile(mode='rb',filetypes=supported_types)

        # user picks a file which is added to the data dictionary of the APP
        if not file is None:
            APP_REFERENCE.data['loaded_files'].append(ImportedFile(file.name,file.raw.read()))
            #print(len(APP_REFERENCE.data['loaded_files'][file.name].lines))
        else:
            print("ope")
        if not work_block.interview_container is None:
            print('INSERTING')
            work_block.interview_container.configure(state="normal")

            work_block.interview_container.delete('0.0',tkinter.END)
            text = APP_REFERENCE.data['loaded_files'][-1].contents_as_rb.decode()
            work_block.interview_container.insert(tkinter.END,f"{text}\n")
            work_block.interview_container.configure(state="disabled")
        else:
            print("oh no")


    # this method will be used to export the
    # results of our NLP magic post-processing
    # of the data
    @staticmethod
    def export_file(APP_REFERENCE):
        # method currently does nothing...
        print("exporting! - (nothing to export yet....)")


    # Save the file dictionary to a file that
    # can be imported at a later time into the
    # GUI APP
    @staticmethod
    def save_session(APP_REFERENCE):
        filepaths = []
        c_as_rb = [] 

        for f in APP_REFERENCE.data['loaded_files']:
            filepaths.append(f.filepath)
            c_as_rb.append(f.contents_as_text)

        save_dump = {   'settings' : APP_REFERENCE.settings,
                        'fp' : filepaths,'rb' : c_as_rb,
                        #'viewports': APP_REFERENCE.viewports
                        }

        save = open('session.tmp','w')
        save.write(json.dumps(save_dump))


    # Recover the file dictionary to rebuild
    # the most recent file dictionary for the
    # GUI APP. Will always look for 'gui_sess.tmp'
    @staticmethod
    def load_session(APP_REFERENCE):
        saved_state = open('session.tmp','r').read()
        save_data = json.loads(saved_state)

        fp = save_data['fp']
        rb = save_data['rb']
        
        for f,r in zip(fp,rb):
            APP_REFERENCE.data['loaded_files'].append(ImportedFile(f,None,txt=r))
        APP_REFERENCE.settings = save_data['settings']




from nltk.corpus import stopwords
import re
class Algorithms:

    @staticmethod
    def remove_stopwords(sentence, pos_tag_list):
        stop_words = stopwords.words('english')
        stop_words.append("unintelligible")
        stop_words.append("yeah")
        stop_words.append("okay")
        stop_words.append("yes")
        stop_words.append("right")
        stop_words.append("interviewer")
        stop_words.append("interview")
        stop_words.append("record")
        stop_words.append("participant")
        stop_words.append("really")
        stop_words.append("think")
        stop_words.append("well")
        stop_words.append("around")
        stop_words.append("also")
        stop_words.append("like")
        stop_words.append("recording")

        word_list = []
        if pos_tag_list != []:
            for token in nlp(sentence):
                if token.pos_ in pos_tag_list:
                    word_list.append(token.text)
        else:
            word_list = sentence.split()
        new_word_list = word_list.copy()
        for word in list(word_list):
            word_lower = word.lower()
            if word_lower in stop_words or len(word) < 4 or '’' in word:
                new_word_list = list(filter((word).__ne__, new_word_list))
        new_sentence = " ".join(word for word in new_word_list)
        return new_sentence

    # Takes a string and returns an cleansed string
    @staticmethod
    def cleantextstring(text, pos_tag_list=[]):
        text = text.lower()
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''''”''“''…'
        for char in text:
            if char in punc:
                text = text.replace(char, "")
        text = Algorithms.remove_stopwords(text, pos_tag_list)
        return text

    @staticmethod
    # Choose how to classify data 
    def classifier_start(APP_REF):
        root = tkinter.Tk()

    # JENNY'S EMBEDDINGS   
    class BertEmbed:
        def __init__(self):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)


        #returns bert embeddings of a sentence
        def sentenceEmbedding(self,sentence):
            inputs = self.tokenizer(sentence, return_tensors = 'pt')
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            return list(last_hidden_states[0][0])

        #param=[[home/file1,home/file2],[home/file3,home/file4]] where it has category 1 and 2
        #these have list of full filepaths
        def dataExtract(self,filepathlist_list):
            nltk.download('stopwords')
            dataDict = {}
            for i, filepathlist in enumerate(filepathlist_list):
                dataDict["category"+str(i+1)] = []
                for filepath in filepathlist:

                    with open(filepath,'rb') as f:
                        f_info = f.read().decode().split("\n")
                        f_info = list(filter(None, f_info))
                        #<add processing here>
                        for line in f_info:
                            dataDict["category"+str(i+1)].append(Algorithms.cleantextstring(line))
            return dataDict

        def makeFeature(self,line):
            line = re.sub(r"\#|\?|\*|\\n|\,|\.", "", line)
            tensors = self.sentenceEmbedding(line) #embed the paragraph

            #Make features for line
            features = []
            for t in tensors:
                features.append(float(t))
            if len(features)!=768:
                print('bad embedding')
                return -1
            x = self.sentiment_scores(Algorithms.cleantextstring(line)) #add vader sentiments
            y = [float(x['neg']), float(x['neu']), float(x['pos'])]
            features = features+y

            if len(features) == 771:
                return features
            else:
                print("vader error")
                return -1

        #takes the dictionary of our data and then vectorizes them using a transformer
        def bertFromDict(self,category_line_list):
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
                        features = self.makeFeature(line)
                        if not features == -1:
                            featuresList.append(features)
                else:
                    features = self.makeFeature(line)
                    if not features == -1:
                        featuresList.append(features)

            #store results of the embeddings into a file corresponding to the category
            filename = self.category +"embeddings.txt"
            with open(filename, "w") as f:
                wr = csv.writer(f)
                wr.writerows(featuresList)

        #this function takes a sentence and then performs a sentiment analysis using a twitter vader sentiment tool
        def sentiment_scores(self,sentence):
            #create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()

            #the SentimentIntensityAnalyzer object method gives a sentiment dictionary
            #which contains pos, neg, neu, and compound scores.
            sentiment_dict = sid_obj.polarity_scores(sentence)
            return sentiment_dict

        def run(self,APP_REF):
            pop_up = tkinter.Tk()
            text = tkinter.Label(pop_up,text="Num Categories:",width=12,height=2)
            text.pack()
            val = tkinter.Entry(pop_up) 
            val.pack()
            submit = tkinter.Button(pop_up,text="ok",command = lambda : self.import_val(val.get(),pop_up),width=12,height=2)
            file_select = tkinter.Button(pop_up,text='File To Classify',command = lambda : self.get_file_to_classify(),width=12,height=2)
            file_select.pack()
            submit.pack()
            pop_up.mainloop()


            self.file_paths= []
            #popus to get all info 
            for i in range(self.BERT_categories):
                self.category_files = []
                pop_up = tkinter.Tk()
                submit = tkinter.Button(pop_up,text=f"select category{i+1} files",command = lambda : self.get_cat_files(i,pop_up),width=12,height=2)
                submit.pack()
                pop_up.mainloop()
                self.file_paths.append(self.category_files)
            dataDict = APP_REF.data['models']['bert'].dataExtract(self.file_paths)

            for i in range(self.BERT_categories):
                APP_REF.data['models']['bert'].category = str(i+1)
                APP_REF.data['models']['bert'].bertFromDict(dataDict[f"category{i+1}"])
            print('DONE RUNNING')

        def import_val(self,n,pop_up):
            self.BERT_categories = int(n)
            pop_up.destroy() 
            pop_up.quit()
            print(self.BERT_categories)

        def get_file_to_classify(self):
            file = askopenfile(mode='rb',filetypes=Utilities.supported_types)
            self.file_to_classify = ImportedFile(file.name,file.raw.read())

        def get_cat_files(self,i,pop_up):
            files = askopenfiles(mode='rb',filetypes=Utilities.supported_types)
            i_files = []
            for f in files:
                i_files.append(f.name)
            print(i_files)
            self.category_files = i_files
            pop_up.destroy() 
            pop_up.quit()


    # JENNY'S CLASSIFIER  
    class Classifier:

        def __init__(self,APP_REF):
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
            self.app    = APP_REF
        def readFileSimple(self,filepath_list):
            featureList_list = []
            for filepath in filepath_list:
                with open(filepath, 'r') as read:
                    readerStore = csv.reader(read)
                    featureList = list(readerStore)
                    featureList_list.append(featureList)
            return featureList_list

        #trains the model via a logistic regression
        def logReg(self,featureList_list): #takes in list of BERT embeddings of suicidal and non suicidal
            #citation: code below is mostly from https://www.marktechpost.com/2021/02/12/logistic-regression-with-a-real-world-example-in-python/
            X = []
            y = []
            category_num = len(featureList_list)
            for num, featureList in enumerate(featureList_list):
                for i in range(len(featureList)):
                    if not len(featureList[i]) == 0:
                        X.append(featureList[i])
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
        def bertFromLine(self,line):
            tensors = self.app.data['models']['bert'].sentenceEmbedding(line)
            features = []
            for t in tensors:
                features.append(float(t))
            if len(features)!=768:
                print('bad embedding')
                return
            #x = sentiment_scores(clean(i))
            x = self.app.data['models']['bert'].sentiment_scores(line)
            y = [float(x['neg']), float(x['neu']), float(x['pos'])]
            features = features+y
            return features

        def classify_predictforline(self,line, category_num_index, filewriter, classifier, flag): #categorynum is indexed at 0 so category 1 is 0
            features = self.bertFromLine(line)
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


        # Filepath: file we want to classify 
        # Category_num: The cat that we want to check against -- but if verbose: this doesnt makee 
        def classify(self,filepath, category_num=1, outfile_path="classify_highprobabilitylines.txt", flag=""):
            category_num_index = category_num - 1
            classifier = pickle.load(open('classify_classifier.sav', 'rb'))


            #remove file if exists
            if os.path.exists(outfile_path):
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
                            probability = self.classify_predictforline(line, category_num_index, filewriter, classifier, flag)
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
                        probability = self.classify_predictforline(line, category_num_index, filewriter, classifier, flag)
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

        def run(self,APP_REF):
            # Bert is run
            APP_REF.data['models']['bert'].run(APP_REF)

            # classify
            basepath = os.getcwd()
            numdirs = 2
            
            #run logistic regression tests for evaluation, does not have to run right after bertDicSimpler
            embeddingspaths = []
            for i in range(numdirs):
                embeddingspaths.append(str(i+1)+"embeddings.txt")
            featureList_list = self.readFileSimple(embeddingspaths)
            self.logReg(featureList_list)
            filepath = APP_REF.data['models']['bert'].file_to_classify.filepath
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            model = BertModel.from_pretrained('bert-base-uncased', config=config)
            category_num = 2 #which category do you want to look at?
            #optional if set flag to "v" then print out all categories
            self.classify(filepath, category_num, flag="v")



        def import_val(self,n,pop_up):
            self.BERT_categories = int(n)
            pop_up.destroy() 
            pop_up.quit()
            print(self.BERT_categories)

        def get_file_to_classify(self):
            file = askopenfile(mode='rb',filetypes=Utilities.supported_types)
            self.file_to_classify = ImportedFile(file.name,file.raw.read())

        def get_cat_files(self,i,pop_up):
            files = askopenfiles(mode='rb',filetypes=Utilities.supported_types)
            i_files = []
            for f in files:
                i_files.append(f.name)
            self.category_files = i_files
            pop_up.destroy() 
            pop_up.quit()


    class TopicModeler:
        def __init__(self,model_alg):
            self.model_alg =  model_alg # 'Sparse' or 'Multi'

        #Updated on 21MAR
        def sparseldamatrix_topics(self,filepath_list, num_topics, outfilename="sparseldamatrix_topics.txt", pos_tag_list=[], data_words=[]):
            #Create a dicitonary that maps the document name to the list of important words
            file_dict = {}

            if data_words == []:
                for filepath in filepath_list:
                    with open(filepath) as f:
                        list_of_sentences = f.readlines()
                        #new_list_of_sentences = [] #remove
                        word_list = []
                        for sentence in list_of_sentences:
                            if sentence == "\n":
                                continue
                            new_sentence = Algorithms.cleantextstring(sentence, pos_tag_list)
                            new_sentence = re.sub("\n", " ", new_sentence)
                            if new_sentence != "" and new_sentence != " ":
                                #new_list_of_sentences.append(new_sentence) #remove
                                small_word_list = new_sentence.split()
                                for word in small_word_list:
                                    word_list.append(word)
                        #file_dict[file] = new_list_of_sentences
                        file_dict[filepath] = word_list
            else:
                for index in range(len(data_words)):
                    file_dict[f"question{index}"] = data_words[index]

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

            if os.path.exists(outfilename):
                os.remove(outfilename)

            with open (outfilename, "w") as writer:
                for i, topic_dist in enumerate(topic_word):
                    writer.write(f"Topic {i}:\n")
                    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_topics+1):-1]
                    writer.write(str(topic_words) + "\n")

        #Uses LDA Multi-Core Topic Modeling to print out specified topics and a specific number of words from that topic to a specifiable file
        #Updated to unigrams on 21MAR
        def ldamulticoretopics(self,filepath_list, num_topics=10, num_words=3, outfile_path="ldamulticoretopics_out.txt", pos_tag_list=[], data_words=[]):
            if data_words == []:
                whole_text = ""
                try:
                    for filepath in filepath_list:
                        with open(filepath) as f:
                            whole_text = whole_text + " " + f.read()
                except Exception as e :
                    print(e)
                    print("ERROR in ldamulticoretopics")
                    exit(2)
                #split by end punctuation into sentences
                sentences_list = re.split('[\.\?\!\\n]\s*', whole_text)

                #create list of word lists
                for sentence in sentences_list:
                    sentence = Algorithms.cleantextstring(sentence, pos_tag_list)
                    word_list = sentence.split()
                    if sentence != "":
                        data_words.append(word_list)

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
            if os.path.exists(outfile_path):
                os.remove(outfile_path)
            for tuple in topics:
                with open(outfile_path, "a") as w:
                    w.write(f"Category {str(tuple[0])}: {tuple[1]}\n\n")

        def run(self,APP_REF):

            print("run was caled")
            if self.model_alg == 'Multi':
                print("runniong")
                filepath_list = [f.filepath for f in APP_REF.data['loaded_files']]
                num_topics = 10
                self.ldamulticoretopics(filepath_list, num_topics=3, outfilename="ldamulticoretopics_out.txt")
                print("done")
            elif self.model_alg == 'Sparse':
                print("running Sparse")
                filepath_list = [f.filepath for f in APP_REF.data['loaded_files']]
                num_topics = 10
                self.sparseldamatrix_topics(filepath_list, num_topics=3,pos_tag_list=["NOUN","ADJ"])

        def import_val(self,n,pop_up):
            self.topics = int(n)
            pop_up.destroy() 
            pop_up.quit()
            print(self.BERT_categories)

        def get_file_to_classify(self):
            file = askopenfile(mode='rb',filetypes=Utilities.supported_types)
            self.file_to_classify = ImportedFile(file.name,file.raw.read())

        def get_cat_files(self,i,pop_up):
            files = APP_REF.data['loaded_files']
            i_files = []
            for f in files:
                i_files.append(f.name)
            print(i_files)
            self.category_files = i_files
            pop_up.destroy() 
            pop_up.quit()



    class DocClusterer:
        def __init__(self,APP_REF):
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
                self.app = APP_REF

            except ValueError:
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

            # EVERETT ADDED 
            writer = open("CLUSTERS.txt",'w')
            for doc in self.cluster_centers:
                # grab top 10 word IDs
                top10_wid = sorted(range(len(doc)), key=lambda sub: doc[sub])[-10:]

                # map wid to actual words
                comn_word_list = []
                for wid in top10_wid:
                    comn_word_list.append(self.vocab_dict[wid])

                # **** PRINTING THE TOP 10 WORDS IN THE CLUSTER ****
                print(f"Cluster {ix}: {comn_word_list}")
                writer.write(f"Cluster {ix}: {comn_word_list}\n")

                ix += 1


                # EVERETT ADDED 
            writer.close()

            print(f'Number of cluster documents: {self.k}')


    class gpt:
        def __init__(self):
            pass

        def interact_model(self,
            model_name='data',
            seed=None,
            nsamples=1,
            batch_size=1,
            length=100,
            temperature=.8,
            top_k=40,
            top_p=1,
            models_dir='models'):

            models_dir = os.path.expanduser(os.path.expandvars(models_dir))
            if batch_size is None:
                batch_size = 1
            assert nsamples % batch_size == 0

            enc = encoder.get_encoder(model_name, models_dir)
            hparams = model.default_hparams()

            input(f"dir: {models_dir} type: {type(models_dir)}")
            input(f"name: {model_name} t : {type(model_name)}")
            with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
                hparams.override_from_dict(json.load(f))

            if length is None:
                length = hparams.n_ctx // 2
            elif length > hparams.n_ctx:
                raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

            with tf.Session(graph=tf.Graph()) as sess:
                context = tf.placeholder(tf.int32, [batch_size, None])
                np.random.seed(seed)
                tf.set_random_seed(seed)
                output = sample.sample_sequence(
                    hparams=hparams, length=length,
                    context=context,
                    batch_size=batch_size,
                    temperature=temperature, top_k=top_k, top_p=top_p
                )

                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
                saver.restore(sess, ckpt)

                while True:
                    raw_text = input("Model prompt >>> ")
                    while not raw_text:
                        print('Prompt should not be empty!')
                        raw_text = input("Model prompt >>> ")
                    context_tokens = enc.encode(raw_text)
                    generated = 0
                    for _ in range(nsamples // batch_size):
                        out = sess.run(output, feed_dict={
                            context: [context_tokens for _ in range(batch_size)]
                        })[:, len(context_tokens):]
                        for i in range(batch_size):
                            generated += 1
                            text = enc.decode(out[i])
                            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                            print(text)
                    print("=" * 80)

        def run(self,APP_REF):
            APP_REF.data['models']['gpt'].interact_model()  




