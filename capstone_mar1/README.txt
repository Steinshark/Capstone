createwordcloud.py:
Description: Takes a list of filepaths like ["category1/doc1.txt", "category1/doc2.txt"]
and creates word clouds from them
def createwordcloud(filepath_list):
adapted from original file: capstone_jan26/try1.py
local dependencies: from cleantextstring import *

cleantextstring.py
Description: Takes a string and returns an cleansed string
def cleantextstring(text):
adapted from original file: capstone_feb28/final.py
original fun names
-def remove_stopwords(sentence)
-def clean_text(text) -- > cleantextstring(text)
notes:
-must uncomment nltk.download('stopwords') for the first time

ldamulticoretopics.py
def ldamulticoretopics(filepath_list, num_topics):
Description: Takes a list of filepaths like ["category1/doc1.txt", "category1/doc2.txt"]
and performs LDA Multi-Core Topic Modeling on it to write topics to a file
adapted from original file: capstone_jan26/try2.py
**** Orig code holds some of the code to the jupiter code

bertembed.py
Main calls: dataExtract(filepathlist_list); bertFromDict(category_line_list)
Additional calls: sentenceEmbedding(sentence); makeFeature(line); sentiment_scores(sentence)
Description:
-dataExtract: Given list of list of files in a category ([[home/file1,home/file2],
[home/file3,home/file4]] where it has category 1 and 2) and stores all the important words
from the files in the directory in a dictionary with the keys corresponding to
the directories
-bertFromDict: User needs to wait for this function call. It takes a value from
the data dictionary output of dataExtract and makes features from them. It
stores the results of the embeddings into a file corresponding to the category.
Ie: produces category1embeddings.txt ...etc
local dependencies: from cleantextstring import *

classify.py
Main calls: logReg(featureList_list); classify(filepath, category_num, flag)
Additional functions: def readFileSimple(filepath_list):,
classify_predictforline(line, category_num_index, filewriter, classifier); bertFromLine(line);
local dependencies: from bertembed import sentenceEmbedding, makeFeature, sentiment_scores
Description: Perform logistic regression on a saved language model (run bertembed.py
before this), and classifies a new file as one of the known categories
There is a verbose mode in the classify flag parameter to print out the probability
of each category for the file. The length of featureList_list must be >=2 to classify.

sparseldamatrix_topics.py
Main call: sparseldamatrix_topics(filepath_list, num_topics)
Description: Uses a sparse matrix to do LDA topic modeling and writes the topics
to the sparseldamatrix_topics.txt file

splitdata.py
Main call: def splitdata(filepathlist_list):
Additional functions: def file_to_clean_list(filepath):
Description: Takes in a list of a list of paths for each category. It prints out
the first 10 lines of an interview and uses the terminal to prompt the user to
clarify who is the interviewer. It assumes the speech is divided every other line.
The interviewer/interviewee name is also stripped off ("A:" or "Interviewer:" etc)
from the beginning if there are names. But the names must be less than 10 characters
long. This is very specific to our data set and does not always work for our
data set either. The lines from each doc is written into the same file. But at the
end of the liens from each file, it writes "~~END~~'{filepath}'~~". This can be
skipped by the parser or can possibly be useful to determine the files. The
user can input a number outside of 0-10 to skip the interview if the format is
incorrect.
