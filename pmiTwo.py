__author__ = 'maruthi'

import sys
import csv
import codecs
import math
import decimal
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords #Stopwords List
import xlsxwriter as xlS

reload(sys)
sys.setdefaultencoding('utf8')
en_sw = set(stopwords.words('english'))

# Variable declaration
row0 = 0
col0 = 0
corpusGl = list()
corpusPos = list()
corpusNeg = list()

# Source connections
# All
dataOpen1 = codecs.open("pmi/foodSentiments.csv", 'rU')
d1 = csv.reader(dataOpen1,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d1.next()
# Food
dataOpen2 = codecs.open("pmi/foodPos.csv", 'rU')
d2 = csv.reader(dataOpen2,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d2.next()
dataOpen3 = codecs.open("pmi/foodNeg.csv", 'rU')
d3 = csv.reader(dataOpen3,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d3.next()
# Ambience
dataOpen4 = codecs.open("pmi/ambPos.csv", 'rU')
d4 = csv.reader(dataOpen4,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d4.next()
dataOpen5 = codecs.open("pmi/ambNeg.csv", 'rU')
d5 = csv.reader(dataOpen5,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d5.next()
# Service
dataOpen6 = codecs.open("pmi/servicePos.csv", 'rU')
d6 = csv.reader(dataOpen6,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d6.next()
dataOpen7 = codecs.open("pmi/serviceNeg.csv", 'rU')
d7 = csv.reader(dataOpen7,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d7.next()
# Price
dataOpen8 = codecs.open("pmi/pricePos.csv", 'rU')
d8 = csv.reader(dataOpen8,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d8.next()
dataOpen9 = codecs.open("pmi/priceNeg.csv", 'rU')
d9 = csv.reader(dataOpen9,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d9.next()

# Output
workbook = xlS.Workbook("markTwoPmiOne.xlsx")
worksheet0 = workbook.add_worksheet('PMI Score')
worksheet0.write(row0, col0, "Term")
worksheet0.write(row0, col0 + 1, "Positive Food PMI Score")
worksheet0.write(row0, col0 + 2, "Negative Food PMI Score")
worksheet0.write(row0, col0 + 3, "Positive Ambience PMI Score")
worksheet0.write(row0, col0 + 4, "Negative Ambience PMI Score")
worksheet0.write(row0, col0 + 5, "Positive Service PMI Score")
worksheet0.write(row0, col0 + 6, "Negative Service PMI Score")
worksheet0.write(row0, col0 + 7, "Positive Price PMI Score")
worksheet0.write(row0, col0 + 8, "Negative Price PMI Score")

# Function for stemming
def stemWord(word):
    stemmer = PorterStemmer()
    stemmedWord = stemmer.stem(word)
    return stemmedWord

def pmi_score(freqWPos, N, freqW, freqPos):
    # PMI Calculation goes here
    x = (decimal.Decimal(freqWPos) * decimal.Decimal(N))/(decimal.Decimal(freqW) * decimal.Decimal(freqPos))
    if x < 0:
        x = x * -1
        pmiScore = math.log(x,2)
        pmiScore = pmiScore * -1
    elif x > 0:
        pmiScore = math.log(x,2)
    else:
        pmiScore = x
    return pmiScore

# Function for preparing bag of words
def textPrep(review):
    preped = list()
    preped[:] = []
    # For getting rid of the stupid issue (Forgot the exact error) This will save you a lot of trouble
    review = unicode(review, errors='ignore')
    review = review.lower().split()
    # Stop word removal, stemming(Porter Stemming) and basic text processing
    nonStopWord = [wordSt for wordSt in review if wordSt not in en_sw]
    for word in nonStopWord:
        word = re.sub(r'[^\w\s]','',word)
        word = re.sub(r'[0-9]','',word)
        word = stemWord(word)
        if word != "":
            preped.append(word)
    return preped


# Inputs for PMI will be preped here
def pmi_input(data):
    corpusGl[:] = []
    # For reviews
    for review in data:
        processedReviewGl = textPrep(review[1])
        # print processedReviewGl
        # print type(processedReviewGl)
        corpusGl.append(processedReviewGl[0])
    # freq(w) / freq(w, pos) / freq(w, neg)
    corpusWordFreqGl = Counter(corpusGl)
    # N / freq(pos) / freq(neg)
    corpusWordCountGl = len(corpusGl)
    return corpusWordFreqGl, corpusWordCountGl

# All
corpusAll = pmi_input(d1)
# freq(w)
corpusWordFreqAll = corpusAll[0]
# N
corpusWordCountAll = corpusAll[1]

# Food
corpusFoodPos = pmi_input(d2)
# freq(w, pos)
corpusWordFreqFoodPos = corpusFoodPos[0]
# freq(pos)
corpusWordCountFoodPos = corpusFoodPos[1]
corpusFoodNeg = pmi_input(d3)
# freq(w, neg)
corpusWordFreqFoodNeg = corpusFoodNeg[0]
# freq(neg)
corpusWordCountFoodNeg = corpusFoodNeg[1]

# Ambience
corpusAmbPos = pmi_input(d4)
# freq(w, pos)
corpusWordFreqAmbPos = corpusAmbPos[0]
# freq(pos)
corpusWordCountAmbPos = corpusAmbPos[1]
corpusAmbNeg = pmi_input(d5)
# freq(w, neg)
corpusWordFreqAmbNeg = corpusAmbNeg[0]
# freq(neg)
corpusWordCountAmbNeg = corpusAmbNeg[1]

# Service
corpusServicePos = pmi_input(d6)
# freq(w, pos)
corpusWordFreqServicePos = corpusServicePos[0]
# freq(pos)
corpusWordCountServicePos = corpusServicePos[1]
corpusServiceNeg = pmi_input(d7)
# freq(w, neg)
corpusWordFreqServiceNeg = corpusServiceNeg[0]
# freq(neg)
corpusWordCountServiceNeg = corpusServiceNeg[1]

# Price
corpusPricePos = pmi_input(d8)
# freq(w, pos)
corpusWordFreqPricePos = corpusPricePos[0]
# freq(pos)
corpusWordCountPricePos = corpusPricePos[1]
corpusPriceNeg = pmi_input(d9)
# freq(w, neg)
corpusWordFreqPriceNeg = corpusPriceNeg[0]
# freq(neg)
corpusWordCountPriceNeg = corpusPriceNeg[1]

for oneWordAll, oneWordCountAll in corpusWordFreqAll.items():
    # Food
    worksheet0.write(row0 + 1, col0, oneWordAll)
    foodPos = dict([ft for ft in corpusWordFreqFoodPos.iteritems() if oneWordAll == ft[0]])
    if len(foodPos) > 0:
        for k,v in foodPos.items():
            oneWordCountFoodPos = v
            pmiPosFoodScore = pmi_score(oneWordCountFoodPos,corpusWordCountAll,oneWordCountAll,corpusWordCountFoodPos)
            worksheet0.write(row0 + 1, col0 + 1, pmiPosFoodScore)
    else:
        pmiPosFoodScore = 0
        worksheet0.write(row0 + 1, col0 + 1, pmiPosFoodScore)

    foodNeg = dict([ft for ft in corpusWordFreqFoodNeg.iteritems() if oneWordAll == ft[0]])
    if len(foodNeg) > 0:
        for k,v in foodNeg.items():
            oneWordCountFoodNeg = v
            pmiNegFoodScore = pmi_score(oneWordCountFoodNeg,corpusWordCountAll,oneWordCountAll,corpusWordCountFoodNeg)
            worksheet0.write(row0 + 1, col0 + 2, pmiNegFoodScore)
    else:
        pmiNegFoodScore = 0
        worksheet0.write(row0 + 1, col0 + 2, pmiNegFoodScore)


    # Ambience
    AmbPos = dict([ft for ft in corpusWordFreqAmbPos.iteritems() if oneWordAll == ft[0]])
    if len(AmbPos) > 0:
        for k,v in AmbPos.items():
            oneWordCountAmbPos = v
            pmiPosAmbScore = pmi_score(oneWordCountAmbPos,corpusWordCountAll,oneWordCountAll,corpusWordCountAmbPos)
            worksheet0.write(row0 + 1, col0 + 3, pmiPosAmbScore)
    else:
        pmiPosAmbScore = 0
        worksheet0.write(row0 + 1, col0 + 3, pmiPosAmbScore)

    AmbNeg = dict([ft for ft in corpusWordFreqAmbNeg.iteritems() if oneWordAll == ft[0]])
    if len(AmbNeg) > 0:
        for k,v in AmbNeg.items():
            oneWordCountAmbNeg = v
            pmiNegAmbScore = pmi_score(oneWordCountAmbNeg,corpusWordCountAll,oneWordCountAll,corpusWordCountAmbNeg)
            worksheet0.write(row0 + 1, col0 + 4, pmiNegAmbScore)
    else:
        pmiNegAmbScore = 0
        worksheet0.write(row0 + 1, col0 + 4, pmiNegAmbScore)

    # Service
    ServicePos = dict([ft for ft in corpusWordFreqServicePos.iteritems() if oneWordAll == ft[0]])
    if len(ServicePos) > 0:
        for k,v in ServicePos.items():
            oneWordCountServicePos = v
            pmiPosServiceScore = pmi_score(oneWordCountServicePos,corpusWordCountAll,oneWordCountAll,corpusWordCountServicePos)
            worksheet0.write(row0 + 1, col0 + 5, pmiPosServiceScore)
    else:
        pmiPosServiceScore = 0
        worksheet0.write(row0 + 1, col0 + 5, pmiPosServiceScore)

    ServiceNeg = dict([ft for ft in corpusWordFreqServiceNeg.iteritems() if oneWordAll == ft[0]])
    if len(ServiceNeg) > 0:
        for k,v in ServiceNeg.items():
            oneWordCountServiceNeg = v
            pmiNegServiceScore = pmi_score(oneWordCountServiceNeg,corpusWordCountAll,oneWordCountAll,corpusWordCountServiceNeg)
            worksheet0.write(row0 + 1, col0 + 6, pmiNegServiceScore)
    else:
        pmiNegServiceScore = 0
        worksheet0.write(row0 + 1, col0 + 6, pmiNegServiceScore)

    # Price
    PricePos = dict([ft for ft in corpusWordFreqPricePos.iteritems() if oneWordAll == ft[0]])
    if len(PricePos) > 0:
        for k,v in PricePos.items():
            oneWordCountPricePos = v
            pmiPosPriceScore = pmi_score(oneWordCountPricePos,corpusWordCountAll,oneWordCountAll,corpusWordCountPricePos)
            worksheet0.write(row0 + 1, col0 + 7, pmiPosPriceScore)
        else:
            pmiPosPriceScore = 0
            worksheet0.write(row0 + 1, col0 + 7, pmiPosPriceScore)

    PriceNeg = dict([ft for ft in corpusWordFreqFoodNeg.iteritems() if oneWordAll == ft[0]])
    if len(PriceNeg) > 0:
        for k,v in PriceNeg.items():
            oneWordCountPriceNeg = v
            pmiNegPriceScore = pmi_score(oneWordCountPriceNeg,corpusWordCountAll,oneWordCountAll,corpusWordCountPriceNeg)
            worksheet0.write(row0 + 1, col0 + 8, pmiNegPriceScore)
    else:
        pmiNegPriceScore = 0
        worksheet0.write(row0 + 1, col0 + 8, pmiNegPriceScore)

    row0 += 1

workbook.close()

print "I can see Timon beyond the horizon!!! HaKuNa mAtAtA!!!"
