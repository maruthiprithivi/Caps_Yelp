__author__ = 'maruthi'

import re
import string
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer #Regular Expression Based Tokenizer
from nltk.corpus import stopwords #Stopwords List
from nltk.corpus import wordnet #WordNet
from nltk.stem import WordNetLemmatizer #Lemmatizer
from nltk.stem.porter import PorterStemmer #Stemmer
from nltk.util import ngrams # N-Grams
"""
# Contraction Expansion List and Function
contractions_dict =  {
    "aren't":"are not",
    "can't":"cannot",
    "couldn't":"could not",
    "didn't":"did not",
    "doesn't":"does not",
    "don't":"do not",
    "hadn't":"had not",
    "hasn't":"has not",
    "haven't":"have not",
    "he'd":"he had",
    "he'll":"he will",
    "he's":"he is",
    "I'd":"I had",
    "I'll":"I will",
    "I'm":"I am",
    "I've":"I have",
    "isn't":"is not",
    "let's":"let us",
    "mightn't":"might not",
    "mustn't":"must not",
    "must've":"must have",
    "shan't":" shall not",
    "she'd":"she would",
    "she'll":"she shall",
    "she's":"she is",
    "shouldn't":" should not",
    "that's":"that is",
    "there's":"there is",
    "they'd":"they had",
    "they'll":"they will",
    "they're":"they are",
    "they've":"they have",
    "we'd":"we had",
    "we're":"we are",
    "we've":"we have",
    "weren't":"were not",
    "what'll":"what will",
    "what're":"what are",
    "what's":"what is",
    "what've":"what have",
    "where's":"where is",
    "who's":"who had",
    "who'll":"who will",
    "who're":"who are",
    "who's":"who is",
    "who've":"who have",
    "won't":"will not",
    "wouldn't":" would not",
    "you'd":"you had",
    "you'll":"you will",
    "you're":"you are",
    "you've":"you have "
  }

 contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
 def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)
"""

regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
wd_lm = WordNetLemmatizer()
pr_sm = PorterStemmer()
dataOpen = open("dataset/Data_101.csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
op1 = csv.writer(open("dataset/review_aspects.csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header = ["Business_Id","Reviewer_Id", "Stars", "Review", "Food", "Ambience", "Service", "Price"]
op1.writerow(header)
## Wrte to txt file
# dR1 = open("dataset/review_aspects01.txt","w")

# print dataOpen
# data1 = dataOpen.read()
# print type(d1)
x = 0
y = 0
cnt = 0
ct = dict()
test = list()
aspAmbi = list()
aspFood = list()
aspServ = list()
aspPric = list()
# aspFood = str()
## Test code for stopwords
"""
words = ["Can't", 'is', 'a', 'contraction']
wrdRM = [word for word in words if word not in en_sw]
print wrdRM
"""
# Used for counting
# for q in d1:
#     ct = q.count(q[12])
#     print q[12],"-",ct
"""
Custom definition to iterate through the broken reviews
"""
def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

fooD = ["food","delicious","yummy", "tasty", "fresh", "salad"]
ambI = ["atmosphere"]
serV = ["staff","looking"]
priC = ["price","cheap","expensive"]
for i in d1:
    y = y + 1
    if y >= 5: break
    """
    The below stretch of codes is to  break and parse
    the statements into silos of 4 major aspects(Food, Ambience, Service, Price)
    pertaining to Restaurants
    """

    txt = i[14].replace("!",".")
    txt = txt.replace(",",".")
    txt = txt.replace("-",".")
    # txt = txt.replace(" ","")
    txtSplit = txt.split(".")
    # wordT for wordT in wordLemma if wordT not in en_sw
    txtL = [wordL.lower() for wordL in txtSplit]
    for e in txtL:
        # print e
        if words_in_string(fooD,e):
            # aspFood =+ e
            aspFood.append(e)
            print "#### Food:",aspFood
        if words_in_string(ambI,e):
            aspAmbi.append(e)
            print "#### Ambience:",aspAmbi
        if words_in_string(serV,e):
            aspServ.append(e)
            print "#### Ambience:",aspAmbi
        if words_in_string(priC,e):
            aspPric.append(e)
            print "#### Ambience:",aspAmbi
    aspFd = aspFood.__str__()
    aspFd = aspFd.replace("[","")
    aspFd = aspFd.replace("]","")
    aspFd = aspFd.replace("'","")
    aspFd = aspFd.replace('"',"'")
    aspFd = aspFd.replace(",","")
    aspAm = aspAmbi.__str__()
    aspAm = aspAm.replace("[","")
    aspAm = aspAm.replace("]","")
    aspAm = aspAm.replace("'","")
    aspAm = aspAm.replace('"',"'")
    aspAm = aspAm.replace(",","")
    aspAm = aspAmbi.__str__()
    aspAm = aspAm.replace("[","")
    aspAm = aspAm.replace("]","")
    aspAm = aspAm.replace("'","")
    aspAm = aspAm.replace('"',"'")
    aspAm = aspAm.replace(",","")
    aspAm = aspAmbi.__str__()
    aspAm = aspAm.replace("[","")
    aspAm = aspAm.replace("]","")
    aspAm = aspAm.replace("'","")
    aspAm = aspAm.replace('"',"'")
    aspAm = aspAm.replace(",","")
    data = i[0], i[12], i[13], i[14], aspFd, aspAm
    # data1 = i[0]+"\t",i[12]+"\t",i[13]+"\t",i[14]+"\t", aspFood+"\t",aspAmbi
    op1.writerow(data)
    print data
    del aspFood[:]
    del aspAmbi[:]





    # NLTK usage starts here!!
"""
for i in d1:
    y = y + 1
    if y >= 5: break
    # wordToken = word_tokenize(i[14]) ## Testing
    wordToken = regToken.tokenize(i[14]) ## Actual Tokenize
    biGrams = ngrams(i,2) #BiGrams
    # for grams in biGrams:
    #     print grams
    gr = biGrams
    wordLow = [wordL.lower() for wordL in wordToken] ##Change to lower case
    # wordUp = [wordL.upper() for wordL in wordToken] ##Change to upper case
    wordLemma = [wd_lm.lemmatize(wordLe) for wordLe in wordLow] ##Run Through Lemmatization
    wordStem = [pr_sm.stem(wordLe) for wordLe in wordLow] ##Run Through Stemming
    wordRemoved = [wordT for wordT in wordLemma if wordT not in en_sw] ##Removing StopWords
    wordPos = nltk.pos_tag(wordRemoved) ##POS Tagging
    # for word in wordPos:
    #     del wordLemPos[:]
    #     wordLemPos = [wd_lm.lemmatize(wordLe) for wordLe in word] ##Run Through Lemmatization
    #     wordLem = wordLemPos.append(wordLemPos)
    # if y > 1:
    """
    # print txtSplit
    # print txtL
    # print tsp
    # print i
    # print gr
    # print wordLemma
    # print wordStem
    # print wordRemoved
    # print wordPos

    #     print wordStem
    # print wordLem
    #     print wordRemoved
    #     print len(wordRemoved)
    #     print wordPos
    #     print len(wordPos)
        # print type(wordPos)
        # print wordLemPos

# print data
print "Going Good"
