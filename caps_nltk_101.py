__author__ = 'maruthi'

import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
# from nltk.tokenize import TreebankWordTokenizer

# tokenizer = TreebankWordTokenizer()
regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
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
    wordLow = [wordL.lower() for wordL in wordToken]
    wordRemoved = [wordT for wordT in wordLow if wordT not in en_sw]
    # wordRemoved = [wordT for wordT in i[14] if wordT not in en_sw]
    wordPos = nltk.pos_tag(wordRemoved)
    if y > 1:
        print wordLow
        print wordRemoved
        print wordPos
        print type(wordPos)

    # print tokenizer(i[14])
    # print type(i)
    # dataRead = i.split(',')
    # cnt = len(i)
    # print dataRead
    # print range(len(dataRead))
    # print type(dataRead)
    # for i,e in dataRead:
    #     print e
    #     x = x + 1
    #     if x > 5: break


print "Going Good"
