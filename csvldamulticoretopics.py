from cleantextstring import *
import csv
from ldamulticoretopics import *

#Given a csv filename, the row the data starts on, the column the responses start on, the parts of speech list it should look at (optional), the name of the output file, the number of topics,
#the number of words in each topic, and the question number of interest (optional, is not zero-index and starts on question 1 on data_row start, if not set, it looks at all the data)
#This function writes to the output file the ldamulticoretopics

def csvldamulticoretopics (filepath, data_row, data_col, outfilename, num_topics, num_words_per_topic, question_num=None, pos_list=[]):
    response_words_list = [] #Create list to hold response words ie: [["hi", "good"], ["happy", "bye"]]
    #Create list to hold question words in the future

    # Open file
    with open(filepath) as file_obj:

        # Create reader object by passing the file
        # object to reader method
        reader_obj = csv.reader(file_obj)

        # Iterate over each row in the csv
        # file using reader object
        for row_index, row in enumerate(reader_obj):
            if row_index < data_row:
                continue #skip header
            response_words_list.append([])
            for index, cell in enumerate(row):
                if index < data_col:#Skip question number
                    continue
                else:
                    #Deal with responses
                    clean_cell_str = cleantextstring(cell, pos_list)
                    if clean_cell_str != "" and clean_cell_str != " ":
                        clean_cell_word_list = clean_cell_str.split()
                        #for word in clean_cell_word_list:
                        list_index = row_index - data_row
                        response_words_list[list_index].append(clean_cell_word_list)

        #Call the ldamulticoretopics function
        data_list = []
        if question_num == None:
            for index in range(len(response_words_list)):
                data_list = data_list + response_words_list[index]
        else:
            data_list = response_words_list[question_num-1]
        if data_list == []:
            print("ERROR: csvldamulticoretopics produced empty data list")
            return False
        ldamulticoretopics(None, num_topics, num_words_per_topic, outfilename, [], data_list)
        return True


filepath = "data.csv"
data_row = 0 #Skip nothing
data_col = 0 #Skip nothing
pos_list = ["NOUN", "ADJ"]
outfilename = "csvldamulticoretopics_out_all.txt"
num_topics = 8
num_words_per_topic = 3
question_num = None
csvldamulticoretopics (filepath, data_row, data_col, outfilename, num_topics, num_words_per_topic, question_num, pos_list)

for index in range(8): #Not including 8
    question_num = index + 1
    num_topics = 5
    num_words_per_topic = 3
    outfilename = f"csvldamulticoretopics_out_question{question_num}.txt"
    csvldamulticoretopics (filepath, data_row, data_col, outfilename, num_topics, num_words_per_topic, question_num, pos_list)
