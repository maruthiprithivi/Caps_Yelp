__author__ = 'maruthi'

import re
import sys
import csv
import codecs
from nltk import pos_tag
from nltk.corpus import stopwords #Stopwords List
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import xlsxwriter as xlS

row0 = 0
col0 = 0
x = 0
y = 0
reload(sys)
sys.setdefaultencoding('utf8')
aspectS = list()
sent_clean = list()
en_sw = set(stopwords.words('english'))

workbook = xlS.Workbook("marker.xlsx")
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

dataOpen = codecs.open("Dataset/Data_101.csv", 'r')
fooD = open("Dataset/food.txt", 'r')
ambiencE = open("Dataset/ambience.txt", 'r')
servicE = open("Dataset/service.txt", 'r')
pricE = open("Dataset/price.txt", 'r')
emotioN = open("Dataset/emotion.txt", 'r')

for revieW in dataOpen:
        if x > 0:
            seQ = 1
            revieW = revieW.split(",")
            print revieW[14]
            sent_tokeN = sent_tokenize(revieW[14])
            y += 1
            print y
            for sentencE in sent_tokeN:
                nouncount = {}
                nounprep = list()
                adjectivecount = {}
                adjectiveprep = list()
                sent_sentimenT = vaderSentiment(sentencE)
                sent_sentiment = { v for k,v in sent_sentimenT.items() if k == "compound"}
                print sent_sentiment
                sent_split = sentencE.decode('utf-8').split()
                tagged_sent = pos_tag(sentencE.decode('utf-8').split())
                sent_cleaN = [wordSt for wordSt in sent_split if wordSt not in en_sw]
                for word in sent_cleaN:
                    word = re.sub(r'[^\w\s]','',word)
                    word = re.sub(r'[0-9]','',word)
                    sent_clean.append(word)
                nounS = [word.lower() for word,pos in tagged_sent if pos == 'NNP']
                for noun in nounS:
                    noun = re.sub(r'[^\w\s]','',noun)
                    noun = re.sub(r'[0-9]','',noun)
                    for word in sent_clean:
                        if word == noun:
                            if word not in nouncount:
                                if word != '':
                                    nouncount[word] = 1
                                else:
                                    continue
                            else:
                                nouncount[word] += 1

                                for k,v in nouncount.items():
                                    prep = k + "(" + str(v) + ")"
                                    nounprep.append(prep)

                    try:
                        if noun in fooD:
                            adjectiveS = [word.lower() for word,pos in tagged_sent if pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
                            for adjectivE in adjectiveS:
                                adjectivE = re.sub(r'[^\w\s]','',adjectivE)
                                adjectivE = re.sub(r'[0-9]','',adjectivE)

                                for word in sent_clean:
                                    if word == adjectivE:
                                        if word in emotioN:
                                            if word not in adjectivecount:
                                                if word != '':
                                                    adjectivecount[word] = 1
                                                else:
                                                    continue
                                            else:
                                                adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotioN:
                                        prep = k + "[" + str(v) + "]"
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        adjectiveprep.append(prep)

                            worksheet0.write(row0 + 1, col0 + 5, "Food_Aspect")
                            worksheet0.write(row0 + 1, col0 + 3, seQ)
                            row0 += 1
                            seQ += 1

                        elif noun in ambiencE:
                            adjectiveS = [word.lower() for word,pos in tagged_sent if pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
                            for adjectivE in adjectiveS:
                                adjectivE = re.sub(r'[^\w\s]','',adjectivE)
                                adjectivE = re.sub(r'[0-9]','',adjectivE)

                                for word in sent_clean:
                                    if word == adjectivE:
                                        if word in emotioN:
                                            if word not in adjectivecount:
                                                if word != '':
                                                    adjectivecount[word] = 1
                                                else:
                                                    continue
                                            else:
                                                adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotioN:
                                        prep = k + "[" + str(v) + "]"
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        adjectiveprep.append(prep)

                            worksheet0.write(row0 + 1, col0 + 6, "Ambience_Aspect")
                            worksheet0.write(row0 + 1, col0 + 3, seQ)
                            row0 += 1
                            seQ += 1

                        elif noun in servicE:
                            adjectiveS = [word.lower() for word,pos in tagged_sent if pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
                            for adjectivE in adjectiveS:
                                adjectivE = re.sub(r'[^\w\s]','',adjectivE)
                                adjectivE = re.sub(r'[0-9]','',adjectivE)

                                for word in sent_clean:
                                    if word == adjectivE:
                                        if word in emotioN:
                                            if word not in adjectivecount:
                                                if word != '':
                                                    adjectivecount[word] = 1
                                                else:
                                                    continue
                                            else:
                                                adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotioN:
                                        prep = k + "[" + str(v) + "]"
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        adjectiveprep.append(prep)

                            worksheet0.write(row0 + 1, col0 + 7, "Service_Aspect")
                            worksheet0.write(row0 + 1, col0 + 3, seQ)
                            row0 += 1
                            seQ += 1

                        elif noun in pricE:
                            adjectiveS = [word.lower() for word,pos in tagged_sent if pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
                            for adjectivE in adjectiveS:
                                adjectivE = re.sub(r'[^\w\s]','',adjectivE)
                                adjectivE = re.sub(r'[0-9]','',adjectivE)

                                for word in sent_clean:
                                    if word == adjectivE:
                                        if word in emotioN:
                                            if word not in adjectivecount:
                                                if word != '':
                                                    adjectivecount[word] = 1
                                                else:
                                                    continue
                                            else:
                                                adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotioN:
                                        prep = k + "[" + str(v) + "]"
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        adjectiveprep.append(prep)

                            worksheet0.write(row0 + 1, col0 + 8, "Price_Aspect")
                            worksheet0.write(row0 + 1, col0 + 3, seQ)
                            row0 += 1
                            seQ += 1

                        else:
                            adjectiveS = [word.lower() for word,pos in tagged_sent if pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
                            for adjectivE in adjectiveS:
                                adjectivE = re.sub(r'[^\w\s]','',adjectivE)
                                adjectivE = re.sub(r'[0-9]','',adjectivE)

                                for word in sent_clean:
                                    if word == adjectivE:
                                        if word in emotioN:
                                            if word not in adjectivecount:
                                                if word != '':
                                                    adjectivecount[word] = 1
                                                else:
                                                    continue
                                            else:
                                                adjectivecount[word] += 1

                                for k,v in adjectivecount.items():
                                    if k in emotioN:
                                        prep = k + "[" + str(v) + "]"
                                        adjectiveprep.append(prep)
                                    else:
                                        prep = k + "(" + str(v) + ")"
                                        adjectiveprep.append(prep)

                            worksheet0.write(row0 + 1, col0 + 9, "Unknown_Aspect")
                            worksheet0.write(row0 + 1, col0 + 3, seQ)
                            row0 += 1
                            seQ += 1

                    except:
                        row0 += 1
                        continue

                worksheet0.write(row0 + 1, col0, revieW[12])
                worksheet0.write(row0 + 1, col0 + 1, revieW[0])
                worksheet0.write(row0 + 1, col0 + 2, revieW[18])


                worksheet0.write(row0 + 1, col0 + 10, "Sentiment")
                worksheet0.write(row0 + 1, col0 + 11, len(sent_clean))


        else:
            x += 1
            continue

workbook.close()

print "Hakuna Matatatatatata!!"
