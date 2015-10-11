__author__ = 'maruthi'

import re
import sys
import csv
import codecs
import time
from Lab3_Yelp2 import *
from nltk import pos_tag
from nltk.corpus import stopwords #Stopwords List
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import xlsxwriter as xlS
i = 0
reviewCount = 100000
row0 = 0
col0 = 0
x = 0
y = 0
foodaspecT = list()
serviceaspecT = list()
ambienceaspecT = list()
priceaspecT = list()
unknownaspecT = list()
nonStopWordBag = list()
reload(sys)
sys.setdefaultencoding('utf8')
aspectS = list()
sent_clean = list()
nounList = list()
adjectivecount = {}
adjectiveprep = list()
en_sw = set(stopwords.words('english'))

workbook = xlS.Workbook("markOne50k.xlsx")
worksheet0 = workbook.add_worksheet('marked')
worksheet0.write(row0, col0, "User_Id")
worksheet0.write(row0, col0 + 1, "Business_Id")
worksheet0.write(row0, col0 + 2, "Date")
worksheet0.write(row0, col0 + 3, "Sequence")
worksheet0.write(row0, col0 + 4, "Sentence")
worksheet0.write(row0, col0 + 5, "Food_Aspect")
worksheet0.write(row0, col0 + 6, "Ambience_Aspect")
worksheet0.write(row0, col0 + 7, "Service_Aspect")
worksheet0.write(row0, col0 + 8, "Price_Aspect")
worksheet0.write(row0, col0 + 9, "Unknown_Aspect")
worksheet0.write(row0, col0 + 10, "Sentiment")
worksheet0.write(row0, col0 + 11, "Count_Of_Non_Stop_Words")
worksheet0.write(row0, col0 + 12, "Review_Id")



splitter = Splitter()
postagger = POSTagger()

dataOpen = codecs.open("mktest2.csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
fooD = open("food1.txt", 'r')
foodFile = fooD.read().split(',')
ambiencE = open("ambience1.txt", 'r')
ambienceFile = ambiencE.read().split(',')
servicE = open("service1.txt", 'r')
serviceFile = servicE.read().split(',')
pricE = open("price1.txt", 'r')
priceFile = pricE.read().split(',')
emotioN = open("emotion1.txt", 'r')
emotionFile = emotioN.read().split(',')

startTime0 = time.time()
for revieW in d1:
        # startTime1 = time.time()

        if x > 0:
            reviewCount += 1
            sentenceLoc = 0
            seQ = 1
            revieW[14] = unicode(revieW[14], errors='ignore')
            sentToken = sent_tokenize(revieW[14])
            sent_tokeN = splitter.split(revieW[14])
            posTagSentence = postagger.pos_tag(sent_tokeN)
            # print posTagSentence
            for sentencE in sent_tokeN:
                if sentencE[0] != '!':
                    y += 1
                    nouncount = {}
                    nounprep = list()
                    # startTime2 = time.time()
                    nonStopWord = [wordSt for wordSt in sentencE if wordSt not in en_sw]
                    for word in nonStopWord:
                        word = re.sub(r'[^\w\s]','',word)
                        word = re.sub(r'[0-9]','',word)
                        if word != "":
                            nonStopWordBag.append(word)
                    sentenceSentiment = " ".join(sentencE)
                    # print sentenceSentiment
                    sent_sentimenT = vaderSentiment(sentenceSentiment)
                    for g,h in sent_sentimenT.items():
                        sent_sentiment = h
                    # endTime2 = time.time()
                    # sentimentRunTime = endTime2 - startTime2
                    # print seQ,".Time taken for vader sentiment to run: ",sentimentRunTime
                    # sent_split = sentencE.decode('utf-8').split()
                    tagSentence = posTagSentence[sentenceLoc]
                    # print tagSentence
                    for tag in tagSentence:
                        for noun in tag[2]:
                            if noun == "NN" or noun == "NNP" or noun == "NNPS" or noun == "NNS" :
                                for word in sent_tokeN[sentenceLoc]:
                                    word = re.sub(r'[^\w\s]','',word)
                                    word = re.sub(r'[0-9]','',word)
                                    if word not in en_sw:
                                        if word == tag[0]:
                                            nounList.append(word)

                                            if word not in nouncount:
                                                if word != '':
                                                    nouncount[word] = 1
                                                 # else:
                                                #     continue
                                            else:
                                                nouncount[word] += 1

                    for k,v in nouncount.items():
                        prep = k + "(" + str(v) + ")"
                        prep = prep.encode('ascii','ignore')
                        nounprep.append(prep)
                    # print y, nounprep
                    # print nounList
                    for word in nounList:
                        if word in foodFile:
                            for tag in tagSentence:
                                for adjective in tag[2]:
                                    if adjective == "JJ" or adjective == "JJS" or adjective == "JJR" :
                                        for word in sent_tokeN[sentenceLoc]:
                                            word = re.sub(r'[^\w\s]','',word)
                                            word = re.sub(r'[0-9]','',word)
                                            if word not in en_sw:
                                                if word == tag[0]:
                                                    if word not in adjectivecount:
                                                        if word != '':
                                                            adjectivecount[word] = 1
                                                         # else:
                                                        #     continue
                                                    else:
                                                        adjectivecount[word] += 1

                            for k,v in adjectivecount.items():
                                if k in emotionFile:
                                    prep = k + "[" + str(v) + "]"
                                    prep = prep.encode('ascii','ignore')
                                    adjectiveprep.append(prep)
                                else:
                                    prep = k + "(" + str(v) + ")"
                                    prep = prep.encode('ascii','ignore')
                                    adjectiveprep.append(prep)
                            if len(adjectiveprep) > 0:
                                foodaspecT = nounprep + adjectiveprep
                            else:
                                foodaspecT = nounprep
                            worksheet0.write(row0 + 1, col0 + 5, ', '.join(foodaspecT))
                    if len(foodaspecT) == 0:
                        for word in nounList:
                            if word in ambienceFile:
                                for tag in tagSentence:
                                    for adjective in tag[2]:
                                        if adjective == "JJ" or adjective == "JJS" or adjective == "JJR" :
                                            for word in sent_tokeN[sentenceLoc]:
                                                word = re.sub(r'[^\w\s]','',word)
                                                word = re.sub(r'[0-9]','',word)
                                                if word not in en_sw:
                                                    if word == tag[0]:
                                                        if word not in adjectivecount:
                                                            if word != '':
                                                                adjectivecount[word] = 1
                                                             # else:
                                                            #     continue
                                                        else:
                                                            adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotionFile:
                                        prep = k + "[" + str(v) + "]"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                if len(adjectiveprep) > 0:
                                    ambienceaspecT = nounprep + adjectiveprep
                                else:
                                    ambienceaspecT = nounprep
                                worksheet0.write(row0 + 1, col0 + 6, ', '.join(ambienceaspecT))
                    if len(ambienceaspecT) == 0:
                        for word in nounList:
                            if word in serviceFile:
                                for tag in tagSentence:
                                    for adjective in tag[2]:
                                        if adjective == "JJ" or adjective == "JJS" or adjective == "JJR" :
                                            for word in sent_tokeN[sentenceLoc]:
                                                word = re.sub(r'[^\w\s]','',word)
                                                word = re.sub(r'[0-9]','',word)
                                                if word not in en_sw:
                                                    if word == tag[0]:
                                                        if word not in adjectivecount:
                                                            if word != '':
                                                                adjectivecount[word] = 1
                                                             # else:
                                                            #     continue
                                                        else:
                                                            adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotionFile:
                                        prep = k + "[" + str(v) + "]"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                if len(serviceaspecT) > 0:
                                    serviceaspecT = nounprep + adjectiveprep
                                else:
                                    serviceaspecT = nounprep
                                worksheet0.write(row0 + 1, col0 + 7, ', '.join(serviceaspecT))

                    if len(serviceaspecT) == 0:
                        for word in nounList:
                            if word in priceFile:
                                for tag in tagSentence:
                                    for adjective in tag[2]:
                                        if adjective == "JJ" or adjective == "JJS" or adjective == "JJR" :
                                            for word in sent_tokeN[sentenceLoc]:
                                                word = re.sub(r'[^\w\s]','',word)
                                                word = re.sub(r'[0-9]','',word)
                                                if word not in en_sw:
                                                    if word == tag[0]:
                                                        if word not in adjectivecount:
                                                            if word != '':
                                                                adjectivecount[word] = 1
                                                             # else:
                                                            #     continue
                                                        else:
                                                            adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotionFile:
                                        prep = k + "[" + str(v) + "]"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        prep = prep.encode('ascii','ignore')
                                        adjectiveprep.append(prep)
                                if len(adjectiveprep) > 0:
                                    priceaspecT = nounprep + adjectiveprep
                                else:
                                    priceaspecT = nounprep
                                worksheet0.write(row0 + 1, col0 + 8, ''.join(priceaspecT))
                    if len(priceaspecT) < 1 and len(ambienceaspecT) < 1 and len(serviceaspecT) < 1 and len(foodaspecT) < 1:
                        for tag in tagSentence:
                            for adjective in tag[2]:
                                if adjective == "JJ" or adjective == "JJS" or adjective == "JJR" :
                                    for word in sent_tokeN[sentenceLoc]:
                                        word = re.sub(r'[^\w\s]','',word)
                                        word = re.sub(r'[0-9]','',word)
                                        if word not in en_sw:
                                            if word == tag[0]:
                                                if word not in adjectivecount:
                                                    if word != '':
                                                        adjectivecount[word] = 1
                                                     # else:
                                                    #     continue
                                                else:
                                                    adjectivecount[word] += 1

                        for k,v in adjectivecount.items():
                            if k in emotionFile:
                                prep = k + "[" + str(v) + "]"
                                prep = prep.encode('ascii','ignore')
                                adjectiveprep.append(prep)
                            else:
                                prep = k + "(" + str(v) + ")"
                                prep = prep.encode('ascii','ignore')
                                adjectiveprep.append(prep)
                        # print nounprep
                        if len(adjectiveprep) > 0:
                            unknownaspecT = nounprep + adjectiveprep
                        else:
                            unknownaspecT = nounprep
                        worksheet0.write(row0 + 1, col0 + 9, ', '.join(unknownaspecT))

                    worksheet0.write(row0 + 1, col0, revieW[12])
                    worksheet0.write(row0 + 1, col0 + 1, revieW[0])
                    worksheet0.write(row0 + 1, col0 + 2, revieW[18])
                    worksheet0.write(row0 + 1, col0 + 4, sentenceSentiment)
                    worksheet0.write(row0 + 1, col0 + 10, sent_sentiment)
                    worksheet0.write(row0 + 1, col0 + 11, len(nonStopWordBag))
                    worksheet0.write(row0 + 1, col0 + 3, seQ)
                    worksheet0.write(row0 + 1, col0 + 12, reviewCount)
                    adjectiveprep[:] = []
                    adjectivecount.clear()
                    nounprep[:] = []
                    sent_clean[:] = []
                    nouncount.clear()
                    row0 += 1
                    seQ += 1
                    nounList[:] = []
                    nonStopWordBag[:] = []
                    sentenceLoc += 1
                    foodaspecT[:] = []
                    ambienceaspecT[:] = []
                    serviceaspecT[:] = []
                    priceaspecT[:] = []
        else:
            x += 1
            continue

workbook.close()
endTime0 = time.time()
totalRunTime = endTime0 - startTime0

print "Hakuna Matatatatatata done in: ", totalRunTime
