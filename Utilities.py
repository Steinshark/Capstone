# This class is meant to make the GUI_APP class cleaner
# "Utilities" will not be instantiated. All methods are
# static
import os                                           # allows access to filepath
import tkinter
from tkinter.filedialog import askopenfiles         # allows for file interaction
from tkinter.filedialog import askopenfile          # allows for file interaction
from tkinter.filedialog import askdirectory         # allows for file interaction
from json import loads, dumps


# NEEDED FOR MODELS 
import csv
import re
import pickle
from transformers import BertTokenizer, BertModel, BertConfig #need install transformers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #need install vaderSentiment
import os
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

# A clean container for imported files
class ImportedFile:
    def __init__(self,filepath,contents_as_rb):
        self.filepath = filepath
        self.contents_as_rb = contents_as_rb
        self.contents_as_text = contents_as_rb.decode()

        self.lines = [l for l in self.contents_as_text.split('\n') if not l == '' and not l == '\n']
        self.words = [w for w in self.contents_as_text.split(' ')  if not w == '']
        self.chars = [c for c in self.contents_as_text  if not c == ' ' or not c =='']



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
        APP_REFERENCE.live_text.insert(tkinter.END,"Importing Files:\n")
        APP_REFERENCE.live_text.yview(tkinter.END)

        # user picks a file which is added to the data dictionary of the APP
        distinct_files = 0

        if not len(file_list) == 0:
            for file in file_list:
                if not file is None and not file.name in APP_REFERENCE.data['loaded_files'].keys():
                    distinct_files += 1
                    # Add to the running instance's data  dictionary
                    APP_REFERENCE.data['loaded_files'][file.name] = ImportedFile(file.name,file.raw.read())

                    # Print status to the GUI text bar
                    APP_REFERENCE.live_text.delete('0.0', END)

                    APP_REFERENCE.live_text.insert(tkinter.END,"\tfetched: " + str(file.name.split('/')[-1]) + "\n")
                    APP_REFERENCE.live_text.yview(tkinter.END)

            APP_REFERENCE.live_text.insert(tkinter.END,"Imported " + str(distinct_files) + " new files\n\n")
            APP_REFERENCE.live_text.yview(tkinter.END)

            # Save session upon any new uploads
            Utilities.save_session(APP_REFERENCE,False)
            return


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
            APP_REFERENCE.data['loaded_files'][file.name] = ImportedFile(file.name,file.raw.read())
            print(len(APP_REFERENCE.data['loaded_files'][file.name].lines))
        if not work_block.interview_container is None:
            work_block.interview_container.delete('0.0',tkinter.END)
            text = APP_REFERENCE.data['loaded_files'][file.name].contents_as_rb.decode()
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
    def save_session(APP_REFERENCE,verbose):
        file_dict = APP_REFERENCE.data['loaded_files']
        save_file = open("gui_sess.tmp",'w')
        for file in file_dict.values():
            save_file.write(str(file))
        save_file.close()

        if verbose:
            APP_REFERENCE.live_text.insert(tkinter.END,"Saved Session: " + str(len(file_dict)) + " files\n")
            APP_REFERENCE.live_text.yview(tkinter.END)


    # Recover the file dictionary to rebuild
    # the most recent file dictionary for the
    # GUI APP. Will always look for 'gui_sess.tmp'
    @staticmethod
    def import_sessions(APP_REFERENCE):
        sessions = {}
        with open('sessions.txt') as save_states:
            for raw_text in save_states.read():
                return loads(raw_text)



    # Recover the file dictionary to rebuild
    # the most recent file dictionary for the
    # GUI APP. Will always look for 'gui_sess.tmp'
    @staticmethod
    def save_session(APP_REFERENCE):
        sessions = {}
        with open('sessions.txt') as save_states:
            for line in save_states.readlines():
                pass


from nltk.corpus import stopwords
import re
class Algorithms:

    @staticmethod
    # Removes small, common, and words with contractions
    def remove_stopwords(sentence):
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
        stop_words.append("recording")
        word_list = sentence.split()
        new_word_list = word_list.copy()

        for word in list(word_list):
            word_lower = word.lower()
            if word_lower in stop_words or len(word) < 4 or '’' in word:
                new_word_list = list(filter((word).__ne__, new_word_list))
        new_sentence = " ".join(word for word in new_word_list)

        return new_sentence

    @staticmethod
    # Takes a string and returns an cleansed string
    def cleantextstring(text):
        text = text.lower()
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''''”''“''…'
        for char in text:
            if char in punc:
                text = text.replace(char, "")
        text = remove_stopwords(text)
        return text

    @staticmethod
    # Choose how to classify data 
    def classifier_start(APP_REF):
        root = tkinter.Tk()

    # JENNY'S EMBEDDINGS   
    class BertEmbed:
        def __init__(self):
            pass 

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

        def run(APP_REF):
            pop_up = tkinter.Tk()
            pop_up.pack()

            text = Tkinter.Text(pop_up,text="Num Categories:",width=50,height=20)
            text.pack()
            val = Tkinter.Entry(pop_up,width=50,height=20) 
            val.pack()
            pop_up.mainloop()
            
            categories = {}



            cat_1_paths = Utilities.import_category_files()

    # JENNY'S CLASSIFIER  
    class Classifier:

        def __init__(self):
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)

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
        def bertFromLine(self,line):
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

        def classify_predictforline(self,line, category_num_index, filewriter, classifier, flag): #categorynum is indexed at 0 so category 1 is 0
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


        # Filepath: file we want to classify 
        # Category_num: The cat that we want to check against -- but if verbose: this doesnt makee 
        def classify(self,filepath, category_num=1, outfile_path="classify_highprobabilitylines.txt", flag=""):
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
            print(f"reach mainloop")
            pop_up.mainloop()
            print(f"leaft")


            categories = {}
            #popus to get all info 
            for i in self.BERT_categories:
                print("enter ",i)
                self.category_files = []
                pop_up = tkinter.Tk()
                submit = tkinter.Button(pop_up,text=f"select category{i} files",command = lambda : self.get_cat_files(i,pop_up),width=12,height=2)
                submit.pack()
                pop_up.mainloop()

        def import_val(self,n,pop_up):
            self.BERT_categories = n
            pop_up.destroy() 
            pop_up.quit()
            print(self.BERT_categories)

        def get_file_to_classify(self):
            file = askopenfile(mode='rb',filetypes=Utilities.supported_types)
            self.file_to_classify = ImportedFile(file.name,file.raw.read())

        def get_cat_files(self,i,pop_up):
            files = askopenfiles(mode='rb',filetypes=Utilities.supported_types)
            for f in files:
                f = ImportedFile(f.name,f.raw.read())
            print(files)
            self.category_files[i] = files
            pop_up.destroy()


    class TopicModeler:
        def __init__(self,model_alg):
            self.model_alg =  model_alg # 'Sparse' or 'Multi'

        def sparseldamatrix_topics(self,filepath_list, num_topics):

            #Create a dicitonary that maps the document name to the list of important words
            file_dict = {}
            for filepath in filepath_list:
                with open(filepath) as f:
                    list_of_sentences = f.readlines()
                    new_list_of_sentences = []
                    for sentence in list_of_sentences:
                        if sentence == "\n":
                            continue
                        new_sentence = Algorithms.cleantextstring(sentence)
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
                sentence = Algorithms.cleantextstring(sentence)
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
       



            if self.model_alg == 'sparse':
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

            elif self.model_alg == 'multicore':
                filepath_list = ["category1/Copy of #1.Ledford.txt", "category1/Copy of #3.Ledford.txt"]
                num_topics = 10
                ldamulticoretopics(filepath_list, num_topics, 3, "ldamulticoretopics_out.txt")

        def run(APP_REF):
            pass