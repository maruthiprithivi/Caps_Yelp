__author__ = 'maruthi'

# Loading dependencies
import sys
import csv
import codecs
import math
import decimal
from collections import Counter
from nltk.stem.porter import * #Porter stemming function
from nltk.tokenize import RegexpTokenizer #Regular Expression Based Tokenizer
from nltk.corpus import stopwords #Stopwords List
import xlsxwriter as xlS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pandas as pd
import os  # for os.path.basename
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from nltk.tag import pos_tag
from scipy.cluster.hierarchy import ward, dendrogram



MDS()
totalvocabStemmed1 = []
totalvocabTokenized1 = []
totalvocabStemmed2 = []
totalvocabTokenized2 = []
totalvocabStemmed3 = []
totalvocabTokenized3 = []
totalvocabStemmed4 = []
totalvocabTokenized4 = []
totalvocabStemmed5 = []
totalvocabTokenized5 = []
bagOfReviews1 = list()
bagOfReviews2 = list()
bagOfReviews3 = list()
bagOfReviews4 = list()
bagOfReviews5 = list()
businessId1 = list()
businessId2 = list()
businessId3 = list()
businessId4 = list()
businessId5 = list()
state1 = list()
state2 = list()
state3 = list()
state4 = list()
state5 = list()
r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0

reload(sys)
sys.setdefaultencoding('utf8')
en_sw = set(stopwords.words('english'))

# Output
# workbook = xlS.Workbook("markTwoPmiOne-Italian.xlsx")
# worksheet0 = workbook.add_worksheet('PMI Score')
# worksheet0.write(row0, col0, "Term")
# worksheet0.write(row0, col0 + 1, "Positive Food PMI Score")

# Tokenizing expression
regToken = RegexpTokenizer("[\w']+")

# For LDA - No use for now
def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# Function for stemming
def stemWord(word):
    stemmer = PorterStemmer()
    stemmedWord = stemmer.stem(word)
    return stemmedWord

# Function for preparing bag of words
def textPrepStem(review):
    preped = list()
    preped[:] = []
    # For getting rid of the stupid issue (Forgot the exact error) This will save you a lot of trouble
    # review = unicode(review, errors='ignore')
    # Sklearn doesn't seem to support unicode
    review = review.encode('ascii','ignore')
    review = regToken.tokenize(review)
    review = [wordL.lower() for wordL in review]
#     review = review.lower().split()
    # Stop word removal, stemming(Porter Stemming) and basic text processing
    nonStopWord = [wordSt for wordSt in review if wordSt not in en_sw]
    for word in nonStopWord:
        word = re.sub(r'[^\w\s]','',word)
        word = re.sub(r'[0-9]','',word)
        word = stemWord(word)
        if word != "":
            preped.append(word)
    return preped

def textPrep(review):
    preped = list()
    preped[:] = []
    # For getting rid of the stupid issue (Forgot the exact error) This will save you a lot of trouble
    # review = unicode(review, errors='ignore')
    # Sklearn doesn't seem to support unicode
    review = review.encode('ascii','ignore')
    review = regToken.tokenize(review)
    review = [wordL.lower() for wordL in review]
    # Stop word removal, stemming(Porter Stemming) and basic text processing
    nonStopWord = [wordSt for wordSt in review if wordSt not in en_sw]
    for word in nonStopWord:
        word = re.sub(r'[^\w\s]','',word)
        word = re.sub(r'[0-9]','',word)
        if word != "":
            preped.append(word)
    return preped

# Loading data
dataOpen1 = codecs.open("Dataset/Data_101.csv", 'rU')
d1 = csv.reader(dataOpen1,delimiter=',',quotechar='"', dialect=csv.excel_tab)
d1.next()

for i in d1:
    # Slicing the dataset based on the star ratings
    if i[13] == "1":
        bagOfReviews1.append(i[14])
        businessId1.append(i[0])
        state1.append(i[5])
        allwordsStemmed1 = textPrepStem(i[14])
        totalvocabStemmed1.extend(allwordsStemmed1)
        allwordsTokenized1 = textPrep(i[14])
        totalvocabTokenized1.extend(allwordsTokenized1)
        r1 += 1
    elif i[13] == "2":
        bagOfReviews2.append(i[14])
        businessId2.append(i[0])
        state2.append(i[5])
        allwordsStemmed2 = textPrepStem(i[14])
        totalvocabStemmed2.extend(allwordsStemmed2)
        allwordsTokenized2 = textPrep(i[14])
        totalvocabTokenized2.extend(allwordsTokenized2)
        r2 += 1
    elif i[13] == "3":
        bagOfReviews3.append(i[14])
        businessId3.append(i[0])
        state3.append(i[5])
        allwordsStemmed3 = textPrepStem(i[14])
        totalvocabStemmed3.extend(allwordsStemmed3)
        allwordsTokenized3 = textPrep(i[14])
        totalvocabTokenized3.extend(allwordsTokenized3)
        r3 += 1
    elif i[13] == "4":
        bagOfReviews4.append(i[14])
        businessId4.append(i[0])
        state4.append(i[5])
        allwordsStemmed4 = textPrepStem(i[14])
        totalvocabStemmed4.extend(allwordsStemmed4)
        allwordsTokenized4 = textPrep(i[14])
        totalvocabTokenized4.extend(allwordsTokenized4)
        r4 += 1
    elif i[13] == "5":
        bagOfReviews5.append(i[14])
        businessId5.append(i[0])
        state5.append(i[5])
        allwordsStemmed5 = textPrepStem(i[14])
        totalvocabStemmed5.extend(allwordsStemmed5)
        allwordsTokenized5 = textPrep(i[14])
        totalvocabTokenized5.extend(allwordsTokenized5)
        r5 += 1

# print ("1 star rating count %d " % r1)
# print ("2 star rating count %d " % r2)
# print ("3 star rating count %d " % r3)
# print ("4 star rating count %d " % r4)
# print ("5 star rating count %d " % r5)

print totalvocabTokenized1
print totalvocabStemmed1

vocabFrame1 = pd.DataFrame({'words': totalvocabTokenized1}, index = totalvocabStemmed1)
vocabFrame2 = pd.DataFrame({'words': totalvocabTokenized2}, index = totalvocabStemmed2)
vocabFrame3 = pd.DataFrame({'words': totalvocabTokenized3}, index = totalvocabStemmed3)
vocabFrame4 = pd.DataFrame({'words': totalvocabTokenized4}, index = totalvocabStemmed4)
vocabFrame5 = pd.DataFrame({'words': totalvocabTokenized5}, index = totalvocabStemmed5)

tfidfVectorizer = TfidfVectorizer(max_df=0.8, max_features=20000000,
                                 min_df=0.4, stop_words='english',
                                 use_idf=True, tokenizer=textPrepStem, ngram_range=(1,2))

tfidfMatrix1 = tfidfVectorizer.fit_transform(bagOfReviews1)
tfidfMatrix2 = tfidfVectorizer.fit_transform(bagOfReviews2)
tfidfMatrix3 = tfidfVectorizer.fit_transform(bagOfReviews3)
tfidfMatrix4 = tfidfVectorizer.fit_transform(bagOfReviews4)
tfidfMatrix5 = tfidfVectorizer.fit_transform(bagOfReviews5)

# print(tfidfMatrix1.shape)
# print(tfidfMatrix2.shape)
# print(tfidfMatrix3.shape)
# print(tfidfMatrix4.shape)
# print(tfidfMatrix5.shape)

terms = tfidfVectorizer.get_feature_names()

dist1 = 1 - cosine_similarity(tfidfMatrix1)
dist2 = 1 - cosine_similarity(tfidfMatrix2)
dist3 = 1 - cosine_similarity(tfidfMatrix3)
dist4 = 1 - cosine_similarity(tfidfMatrix4)
dist5 = 1 - cosine_similarity(tfidfMatrix5)

numClusters = 5

km = KMeans(n_clusters=numClusters)

km.fit(tfidfMatrix1)
km.fit(tfidfMatrix1)
km.fit(tfidfMatrix1)
km.fit(tfidfMatrix1)
km.fit(tfidfMatrix1)

clusters = km.labels_.tolist()

#joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

oneStarReviews = { 'businessId': businessId1, 'reviews': bagOfReviews1, 'cluster': clusters, 'location': state1}
twoStarReviews = { 'businessId': businessId2, 'reviews': bagOfReviews1, 'cluster': clusters, 'location': state2}
threeStarReviews = { 'businessId': businessId3, 'reviews': bagOfReviews1, 'cluster': clusters, 'location': state3}
fourStarReviews = { 'businessId': businessId4, 'reviews': bagOfReviews1, 'cluster': clusters, 'location': state4}
fiveStarReviews = { 'businessId': businessId5, 'reviews': bagOfReviews1, 'cluster': clusters, 'location': state5}

reviewFrame1 = pd.DataFrame(oneStarReviews, index = [clusters] , columns = ['businessId', 'cluster', 'state'])
reviewFrame2 = pd.DataFrame(oneStarReviews, index = [clusters] , columns = ['businessId', 'cluster', 'state'])
reviewFrame3 = pd.DataFrame(oneStarReviews, index = [clusters] , columns = ['businessId', 'cluster', 'state'])
reviewFrame4 = pd.DataFrame(oneStarReviews, index = [clusters] , columns = ['businessId', 'cluster', 'state'])
reviewFrame5 = pd.DataFrame(oneStarReviews, index = [clusters] , columns = ['businessId', 'cluster', 'state'])

reviewFrame1['cluster'].value_counts()
reviewFrame2['cluster'].value_counts()
reviewFrame3['cluster'].value_counts()
reviewFrame4['cluster'].value_counts()
reviewFrame5['cluster'].value_counts()

groupOne = reviewFrame1['businessId'].groupby(reviewFrame1['cluster'])
groupTwo = reviewFrame2['businessId'].groupby(reviewFrame2['cluster'])
groupThree = reviewFrame3['businessId'].groupby(reviewFrame3['cluster'])
groupFour = reviewFrame4['businessId'].groupby(reviewFrame4['cluster'])
groupFive = reviewFrame5['businessId'].groupby(reviewFrame5['cluster'])

groupOne.mean()
groupTwo.mean()
groupThree.mean()
groupFour.mean()
groupFive.mean()


## Needs editing
print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d words:" % i)
    for ind in order_centroids[i, :6]:
        print(' %s' % vocabFrame1.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
    print()
    print()
    print("Cluster %d titles:" % i)
    for title in vocabFrame1.ix[i]['title'].values.tolist():
        print(' %s, ' % title)
    print()
    print()
