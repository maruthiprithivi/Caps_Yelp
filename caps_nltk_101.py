__author__ = 'maruthi'

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer #Regular Expression Based Tokenizer
from nltk.corpus import stopwords #Stopwords List
from nltk.corpus import wordnet #WordNet
from nltk.stem import WordNetLemmatizer #Lemmatizer
from nltk.stem.porter import PorterStemmer #Stemmer


regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
wd_lm = WordNetLemmatizer()
pr_sm = PorterStemmer()
dataOpen = open("dataset/Data_101.csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
# print dataOpen
# data1 = dataOpen.read()
# print type(d1)
x = 0
y = 0
cnt = 0

## Test code for stopwords
"""
words = ["Can't", 'is', 'a', 'contraction']
wrdRM = [word for word in words if word not in en_sw]
print wrdRM
"""
for i in d1:
    y = y + 1
    if y >= 3: break
    # wordToken = word_tokenize(i[14]) ## Testing
    wordToken = regToken.tokenize(i[14]) ## Actual
    wordLow = [wordL.lower() for wordL in wordToken] ##Change to lower case
    wordUp = [wordL.upper() for wordL in wordToken] ##Change to upper case
    wordLem = [wd_lm.lemmatize(wordLe) for wordLe in wordLow] ##Run Through Lemmatization
    wordStem = [pr_sm.stem(wordLe) for wordLe in wordLow] ##Run Through Stemming
    wordRemoved = [wordT for wordT in wordLem if wordT not in en_sw] ##Removing StopWords
    # wordRemoved = [wordT for wordT in i[14] if wordT not in en_sw]
    wordPos = nltk.pos_tag(wordRemoved) ##POS Tagging
    if y > 1:
        print wordLow
        print wordStem
        print wordLem
        print wordRemoved
        # print wordPos
        # print type(wordPos)



print "Going Good"
