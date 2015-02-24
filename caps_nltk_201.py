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
from textblob import TextBlob

# contractions_dict =  {
#     "aren't":"are not",
#     "can't":"cannot",
#     "couldn't":"could not",
#     "didn't":"did not",
#     "doesn't":"does not",
#     "don't":"do not",
#     "hadn't":"had not",
#     "hasn't":"has not",
#     "haven't":"have not",
#     "he'd":"he had",
#     "he'll":"he will",
#     "he's":"he is",
#     "I'd":"I had",
#     "I'll":"I will",
#     "I'm":"I am",
#     "I've":"I have",
#     "isn't":"is not",
#     "let's":"let us",
#     "mightn't":"might not",
#     "mustn't":"must not",
#     "must've":"must have",
#     "shan't":" shall not",
#     "she'd":"she would",
#     "she'll":"she shall",
#     "she's":"she is",
#     "shouldn't":" should not",
#     "that's":"that is",
#     "there's":"there is",
#     "they'd":"they had",
#     "they'll":"they will",
#     "they're":"they are",
#     "they've":"they have",
#     "we'd":"we had",
#     "we're":"we are",
#     "we've":"we have",
#     "weren't":"were not",
#     "what'll":"what will",
#     "what're":"what are",
#     "what's":"what is",
#     "what've":"what have",
#     "where's":"where is",
#     "who's":"who had",
#     "who'll":"who will",
#     "who're":"who are",
#     "who's":"who is",
#     "who've":"who have",
#     "won't":"will not",
#     "wouldn't":" would not",
#     "you'd":"you had",
#     "you'll":"you will",
#     "you're":"you are",
#     "you've":"you have "
#   }
#
#  contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
#  def expand_contractions(s, contractions_dict=contractions_dict):
#      def replace(match):
#          return contractions_dict[match.group(0)]
#      return contractions_re.sub(replace, s)



regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
dataOpen = open("dataset/review_aspect(labeled).csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')

op1 = csv.writer(open("dataset/AspectsSentimentSample(200).csv", "wb"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, escapechar = ",")
header1 = ["Review","Food","Food_Sentiment","Ambience","Ambience_Sentiment","Service","Service_Sentiment", "Price","Price_Sentiment"]
op1.writerow(header1)

# xr = list()
cnt = 0
for i in d1:
    print i[0]
    # xr = i[0]
    # print type(xr)
    cnt = cnt + 1
    # iR1 = re.sub(r'[^\w\s]','',i[1])
    # iR1 = re.sub(r'[0-9]','',iR1)
    wdTk1 = regToken.tokenize(i[1])
    txtL1 = [wordL.lower() for wordL in wdTk1]

    wdRm1 = [wordT for wordT in txtL1 if wordT not in en_sw]
    t1 = wdRm1.__str__()
    t1 = t1.replace("[","")
    t1 = t1.replace("]","")
    t1 = t1.replace("'","")
    t1 = t1.replace('"',"'")
    t1 = t1.replace(",","")
    texT1 = TextBlob(t1)
    senT1 = ("%.2f" % texT1.sentiment.polarity)

    # iR2 = re.sub(r'[^\w\s]','',i[2])
    # iR2 = re.sub(r'[0-9]','',iR2)
    wdTk2 = regToken.tokenize(i[2])
    txtL2 = [wordL.lower() for wordL in wdTk2]

    wdRm2 = [wordT for wordT in txtL2 if wordT not in en_sw]
    t2 = wdRm2.__str__()
    t2 = t2.replace("[","")
    t2 = t2.replace("]","")
    t2 = t2.replace("'","")
    t2 = t2.replace('"',"'")
    t2 = t2.replace(",","")
    texT2 = TextBlob(t2)
    senT2 = ("%.2f" % texT2.sentiment.polarity)


    # iR3 = re.sub(r'[^\w\s]','',i[3])
    # iR3 = re.sub(r'[0-9]','',iR3)
    wdTk3 = regToken.tokenize(i[3])
    txtL3 = [wordL.lower() for wordL in wdTk3]

    wdRm3 = [wordT for wordT in txtL3 if wordT not in en_sw]
    t3 = wdRm3.__str__()
    t3 = t3.replace("[","")
    t3 = t3.replace("]","")
    t3 = t3.replace("'","")
    t3 = t3.replace('"',"'")
    t3 = t3.replace(",","")
    texT3 = TextBlob(t3)
    senT3 = ("%.2f" % texT3.sentiment.polarity)

    # iR4 = re.sub(r'[^\w\s]','',i[4])
    # iR4 = re.sub("[^A-Za-z]",'',i[4])
    wdTk4 = regToken.tokenize(i[4])
    txtL4 = [wordL.lower() for wordL in wdTk4]

    wdRm4 = [wordT for wordT in txtL4 if wordT not in en_sw]
    t4 = wdRm4.__str__()
    t4 = t4.replace("[","")
    t4 = t4.replace("]","")
    t4 = t4.replace("'","")
    t4 = t4.replace('"',"'")
    t4 = t4.replace(",","")
    texT4 = TextBlob(t4)
    senT4 = ("%.2f" % texT4.sentiment.polarity)

    wdTk0 = regToken.tokenize(i[0])
    txtL0 = [wordL.lower() for wordL in wdTk0]
    print txtL0

    t0 = txtL0.__str__()
    t0 = t0.replace("[","")
    t0 = t0.replace("]","")
    t0 = t0.replace("'","")
    t0 = t0.replace('"',"'")
    t0 = t0.replace(",","")
    # print ("{0:.2f}".format(round(senT4,2)))
    # print("%.2f" % texT4.sentiment.polarity)
    # print format(senT4,'.2f')

    # print senT1
    # print senT2
    # print senT3
    # print senT4
    data = i[0],i[1],senT1,i[2],senT2,i[3],senT3,i[4],senT4
    # data = t0,t1,senT1,t2,senT2,t3,senT3[3],t4,senT4
    if cnt > 1:
        op1.writerow([t0,t1,senT1,t2,senT2,t3,senT3,t4,senT4])
print cnt


