# Import the wordcloud library
import matplotlib.pyplot as plt   # we had this one before
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from cleantextstring import *

def createwordcloud(filepath_list):
    try:
        whole_text = ""
        for filepath in filepath_list:
            with open(filepath) as f:
                whole_text = whole_text + " " + f.read()
    except:
        print("ERROR in createwordcloud: Could not open file", filepath)
        exit(2)
    whole_text = cleantextstring(whole_text)
    whole_text = whole_text.replace('\n', ' ')

    # The cloud!
    cloud = WordCloud(width=480, height=480, margin=0).generate(whole_text)    # 'whole_text' is the constructed tweet string

    # Now popup the display of our generated cloud image.
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()

filepath_list = ["category1/Copy of #1.Ledford.txt", "category1/Copy of #3.Ledford.txt"]
createwordcloud(filepath_list)
