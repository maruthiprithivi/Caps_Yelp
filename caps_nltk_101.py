__author__ = 'maruthi'

import re
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
cont_dict =  {
    "aren't":" are not",
    "can't":" cannot",
    "couldn't":" could not",
    "didn't":" did not",
    "doesn't":" does not",
    "don't":" do not",
    "hadn't":" had not",
    "hasn't":" has not",
    "haven't":" have not",
    "he'd":" he had; he would",
    "he'll":" he will; he shall",
    "he's":" he is; he has",
    "I'd":" I had; I would",
    "I'll":" I will; I shall",
    "I'm":" I am",
    "I've":" I have",
    "isn't":" is not",
    "let's":" let us",
    "mightn't":"might not",
    "mustn't":" must not",
    "shan't":" shall not",
    "she'd":" she had; she would",
    "she'll":" she will; she shall",
    "she's":" she is; she has",
    "shouldn't":" should not",
    "that's":" that is; that has",
    "there's":" there is; there has",
    "they'd":" they had; they would",
    "they'll":" they will; they shall",
    "they're":" they are",
    "they've":" they have",
    "we'd":" we had; we would",
    "we're":" we are",
    "we've":" we have",
    "weren't":" were not",
    "what'll":" what will; what shall",
    "what're":" what are",
    "what's":" what is; what has",
    "what've":" what have",
    "where's":" where is; where has",
    "who's":" who had; who would",
    "who'll":" who will; who shall",
    "who're":" who are",
    "who's":" who is; who has",
    "who've":" who have",
    "won't":" will not",
    "wouldn't":" would not",
    "you'd":" you had; you would",
    "you'll":" you will; you shall",
    "you're":" you are",
    "you've":" you have "
  }

cont_re = re.compile('(%s)' % '|'.join(cont_dict.keys()))
def exp_cont(s, cont_dict=cont_dict):
    def replace(match):
        return cont_dict[match.group(0)]
    return cont_re.sub(replace, s)
"""

regToken = RegexpTokenizer("[\w']+")
en_sw = set(stopwords.words('english'))
wd_lm = WordNetLemmatizer()
pr_sm = PorterStemmer()
dataOpen = open("dataset/Data_101.csv", 'r')
d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
op1 = csv.writer(open("dataset/review_aspects.csv", "wb"), delimiter=',', quotechar="", quoting=csv.QUOTE_NONE, escapechar = ",")
header = ["Business_Id","Reviewer_Id", "Review", "Food", "Ambience", "Service", "Price"]
op1.writerow(header)
# print dataOpen
# data1 = dataOpen.read()
# print type(d1)
x = 0
y = 0
cnt = 0
ct = dict()
test = list()
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
fooD = ["food","delicious","yummy", "tasty", "fresh", "salad"]
ambI = ["atmosphere"]
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

    txtL = [wordL.lower() for wordL in txtSplit]
    for x in txtL:
        if x in fooD:
            aspFood = x
            continue
        if x in ambI:
            aspAmbi = x
            continue
        data = [i]
        op1.writerow(data)






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


print "Going Good"
