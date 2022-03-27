#nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import spacy
nlp = spacy.load('en_core_web_sm') #you can use other methods

# Removes small, common, and words with contractions
# Updated on 21MAR
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
def cleantextstring(text, pos_tag_list=[]):
    text = text.lower()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''''”''“''…'
    for char in text:
        if char in punc:
            text = text.replace(char, "")
    text = remove_stopwords(text, pos_tag_list)
    return text
