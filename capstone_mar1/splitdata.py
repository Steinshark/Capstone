import os #for listdir
from os import path #for path
from os.path import exists, isdir

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append("unintelligible")
from cleantextstring import *

#prints error and returns None if file unreadable
def file_to_clean_list(filepath):
    with open(filepath) as f:
        f_info = f.read().split("\n")
        f_info = list(filter(None, f_info))
        counter=0
        for line in f_info:
            if counter > 10:
                break
            print(counter, line)
            counter = counter + 1
        # Fine tunes which lines we are interested in
        interviewer_index = input("Note: interviewer name must be shorter than 10 characters. If format error, put number < 0 or > 10.\nIf want to end program, input 'quit'.\nWho is the interviewer? (input line number) ")
        if interviewer_index == "quit":
            exit(1)
        interviewer_index = int(interviewer_index)
        if interviewer_index < 0 or interviewer_index > 10:
            print("ERROR: There must be something wrong with this file so we will go on to the next one.")
            return None
        interviewee_index = interviewer_index + 1
        print("Then interviewee is", f_info[interviewee_index].partition(":")[0])
        #mode = input("Analyze by \n1)alternating lines of communication or \n2)name of interviewee\nPlease choose a mode: ")
        interviewee_response_list = []
        interviewer_question_list = []
        for line_number, line in enumerate(f_info[interviewer_index:]): #only start from line response starts
            #print("in for with line", line)
            #check to see if need to strip the "A:" off
            if line_number%2 == 1:
                response_tuple = line.partition(":")
                if len(response_tuple[2]) == 0 or len(response_tuple[0]) > 10: # if no A: or if long string then have a :, append the whole line **Assumes the file
                    print("adding", line)
                    interviewee_response_list.append(line)
                else:
                    print("adding2", response_tuple[2].lstrip())
                    interviewee_response_list.append(response_tuple[2].lstrip())
            else: #these are the questions
                question_tuple = line.partition(":")
                if len(question_tuple[2]) == 0 or len(question_tuple[2]) > 10: # if no A: or if long string then have a :, append the whole line **Assumes the file
                    interviewer_question_list.append(line)
                else:
                    interviewer_question_list.append(question_tuple[2].lstrip())
        # Creates list of responses with revelant words and information
        cleaned_interviewee_response_list = []
        cleaned_interviewer_question_list = []
        for response in interviewee_response_list:
            cleaned_response = cleantextstring(response)
            if not cleaned_response == "":
                cleaned_interviewee_response_list.append(cleaned_response)
        for question in interviewer_question_list:
            cleaned_question = cleantextstring(question)
            if not cleaned_question == "":
                cleaned_interviewer_question_list.append(cleaned_question)
        return interviewee_response_list, interviewer_question_list, cleaned_interviewee_response_list, cleaned_interviewer_question_list

def splitdata(filepathlist_list):
    #Remove files from running previously
    file_exists = exists("interviewer_question_list.txt")
    if file_exists:
        os.remove("interviewer_question_list.txt")
    file_exists = exists("interviewee_response_list.txt")
    if file_exists:
        os.remove("interviewee_response_list.txt")
    file_exists = exists("cleaned_interviewer_question_list.txt")
    if file_exists:
        os.remove("cleaned_interviewer_question_list.txt")
    file_exists = exists("cleaned_interviewee_response_list.txt")
    if file_exists:
        os.remove("cleaned_interviewee_response_list.txt")

    #Deal with the data
    for category in filepathlist_list:
        for filepath in category:
            interviewee_response_list, interviewer_question_list, cleaned_interviewee_response_list, cleaned_interviewer_question_list = file_to_clean_list(filepath)
            if cleaned_interviewee_response_list == None: #skip if there's an issue with the file
                continue
            with open("interviewee_response_list.txt", "a") as appender:
                for line in interviewee_response_list:
                    appender.writelines(line+"\n")
                appender.writelines(f"~~END~~'{filepath}'~~")
            with open("interviewer_question_list.txt", "a") as appender:
                for line in interviewer_question_list:
                    appender.writelines(line+"\n")
                appender.writelines(f"~~END~~'{filepath}'~~")
            with open("cleaned_interviewee_response_list.txt", "a") as appender:
                for line in cleaned_interviewee_response_list:
                    appender.writelines(line+"\n")
                appender.writelines(f"~~END~~'{filepath}'~~")
            with open("cleaned_interviewer_question_list.txt", "a") as appender:
                for line in cleaned_interviewer_question_list:
                    appender.writelines(line+"\n")
                appender.writelines(f"~~END~~'{filepath}'~~")

basepath = path.dirname(__file__)
filepathlist_list = []
dir_path = os.getcwd()
for category_dir in os.listdir(dir_path):
    if isdir(category_dir):
        category_dir_path = path.join(dir_path, category_dir)
        filepathlist = []
        for file in os.listdir(category_dir_path):
            filepath = path.join(category_dir_path, file)
            filepathlist.append(filepath)
        filepathlist_list.append(filepathlist)
splitdata(filepathlist_list)
