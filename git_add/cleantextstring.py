#nltk.download('stopwords')
from nltk.corpus import stopwords
import re

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

# Takes a string and returns an cleansed string
def cleantextstring(text):
    text = text.lower()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''''”''“''…'
    for char in text:
        if char in punc:
            text = text.replace(char, "")
    text = remove_stopwords(text)
    return text
