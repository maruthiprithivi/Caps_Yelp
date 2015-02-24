__author__ = 'maruthi'

import re
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer #Regular Expression Based Tokenizer
from nltk.corpus import stopwords #Stopwords List

cont = ["aren't",
"can't",
"couldn't",
"didn't",
"doesn't",
"don't",
"hadn't",
"hasn't",
"haven't",
"he'd",
"he'll",
"he's",
"I'd",
"I'll",
"I'm",
"I've",
"isn't",
"let's",
"mightn't",
"mustn't",
"shan't",
"she'd",
"she'll",
"she's",
"shouldn't",
"that's",
"there's",
"they'd",
"they'll",
"they're",
"they've",
"we'd",
"we're",
"we've",
"weren't",
"what'll",
"what're",
"what's",
"what've",
"where's",
"who's",
"who'll",
"who're",
"who's",
"who've",
"won't",
"wouldn't",
"you'd",
"you'll",
"you're",
"you've",
"it's",
"must've"]

regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
dataOpen = open("dataset/review_aspect(labeled).csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
op1 = csv.writer(open("dataset/aspectsForExtraction(Food).csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header1 = ["Food","Counts"]
op1.writerow(header1)
op2 = csv.writer(open("dataset/aspectsForExtraction(Ambience).csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header2 = ["Ambience","Counts"]
op2.writerow(header2)
op3 = csv.writer(open("dataset/aspectsForExtraction(Service).csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header3 = ["Service","Counts"]
op3.writerow(header3)
op4 = csv.writer(open("dataset/aspectsForExtraction(Price).csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header4 = ["Price","Counts"]
op4.writerow(header4)

aspAmbi = list()
aspFood = list()
aspServ = list()
aspPric = list()
wordcount1 = {}
wordcount2 = {}
wordcount3 = {}
wordcount4 = {}


# s = "string. With. Punctuation?"
# s = re.sub(r'[^\w\s]','',s)

for i in d1:

    # print i[4]

    # iR1 = re.sub(r'[^\w\s]','',i[1])
    # iR1 = re.sub(r'[0-9]','',iR1)
    wdTk1 = regToken.tokenize(i[1])
    txtL1 = [wordL.lower() for wordL in wdTk1]
    txtL1 = [wordT for wordT in txtL1 if wordT not in cont]
    wdRm1 = [wordT for wordT in txtL1 if wordT not in en_sw]

    # iR2 = re.sub(r'[^\w\s]','',i[2])
    # iR2 = re.sub(r'[0-9]','',iR2)
    wdTk2 = regToken.tokenize(i[2])
    txtL2 = [wordL.lower() for wordL in wdTk2]
    txtL2 = [wordT for wordT in txtL2 if wordT not in cont]
    wdRm2 = [wordT for wordT in txtL2 if wordT not in en_sw]

    # iR3 = re.sub(r'[^\w\s]','',i[3])
    # iR3 = re.sub(r'[0-9]','',iR3)
    wdTk3 = regToken.tokenize(i[3])
    txtL3 = [wordL.lower() for wordL in wdTk3]
    txtL3 = [wordT for wordT in txtL3 if wordT not in cont]
    wdRm3 = [wordT for wordT in txtL3 if wordT not in en_sw]

    # iR4 = re.sub(r'[^\w\s]','',i[4])
    # iR4 = re.sub("[^A-Za-z]",'',i[4])
    wdTk4 = regToken.tokenize(i[4])
    txtL4 = [wordL.lower() for wordL in wdTk4]
    txtL4 = [wordT for wordT in txtL4 if wordT not in cont]
    wdRm4 = [wordT for wordT in txtL4 if wordT not in en_sw]

    for e in wdRm1:
        if e not in wordcount1:
            wordcount1[e] = 1
        else:
            wordcount1[e] += 1
        # print type(wordcount1)

    for f in wdRm2:
        if f not in wordcount2:
            wordcount2[f] = 1
        else:
            wordcount2[f] += 1
        # print type(wordcount2)

    for g in wdRm3:
        if g not in wordcount3:
            wordcount3[g] = 1
        else:
            wordcount3[g] += 1
        # print type(wordcount3)

    for h in wdRm4:
        if h not in wordcount4:
            wordcount4[h] = 1
        else:
            wordcount4[h] += 1
        # print type(wordcount4)

for k,v in wordcount1.items():
    data1 = k, v
    op1.writerow(data1)

for k,v in wordcount2.items():
    data2 = k, v
    op2.writerow(data2)

for k,v in wordcount3.items():
    data3 = k, v
    op3.writerow(data3)

for k,v in wordcount4.items():
    data4 = k, v
    op4.writerow(data4)

