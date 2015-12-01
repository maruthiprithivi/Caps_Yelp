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
# import xlsxwriter as xlS
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
import io
import mpld3 as mpld3
from gensim import corpora, models, similarities
import numpy as np
 
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
ranks1 = []
ranks2 = []
ranks3 = []
ranks4 = []
ranks5 = []
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
businessName1 = list()
businessName2 = list()
businessName3 = list()
businessName4 = list()
businessName5 = list()
userId1 = list()
userId2 = list()
userId3 = list()
userId4 = list()
userId5 = list()
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
x1 = 6
x2 = 6
x3 = 6
x4 = 6
x5 = 6

# reload(sys)
# sys.setdefaultencoding('utf8')
en_sw = set(stopwords.words('english'))
 
# Output
# workbook = xlS.Workbook("markTwoPmiOne-Italian.xlsx")
# worksheet0 = workbook.add_worksheet('PMI Score')
# worksheet0.write(row0, col0, "Term")
# worksheet0.write(row0, col0 + 1, "Positive Food PMI Score")
 
# Tokenizing expression
regToken = RegexpTokenizer("[\w']+")
 
# For LDA - No use for now
def stripProppersPOS(text):
    review = regToken.tokenize(text)
    review = [wordL.lower() for wordL in review]
    tagged = pos_tag(review) #use NLTK's part of speech tagger
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
 
    # Sklearn doesn't seem to support unicode
    # try:
    #     # review = unicode(review, errors='ignore')
    #     review = review.encode('ascii','ignore')
    #     # review = review.encode('utf-8')
    # except:
    #     print review
    # review = review.encode('ascii','ignore')
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
 
    # Sklearn doesn't seem to support unicode
    # try:
    #     # review = unicode(review, errors='ignore')
    #     review = review.encode('ascii','ignore')
    #     # review = review.encode('utf-8')
    # except:
    #     print review
    # review = review.encode('ascii','ignore')
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
# dataOpen1 = codecs.open("Dataset/Data_101.csv", 'rU')
dataOpen1 = io.open("Dataset/aspectDataMerged.csv", encoding="ISO-8859-1", errors="ignore")
d1 = csv.reader(dataOpen1,delimiter=',',quotechar='"', dialect=csv.excel_tab)
next(d1)
 
for i in d1:
    # Slicing the dataset based on the star ratings
    if i[13] == "1":
        bagOfReviews1.append(i[14])
        businessId1.append(i[0])
        businessName1.append(i[2])
        userId1.append(i[12])
        state1.append(i[5])
        allwordsStemmed1 = textPrepStem(i[14])
        totalvocabStemmed1.extend(allwordsStemmed1)
        allwordsTokenized1 = textPrep(i[14])
        totalvocabTokenized1.extend(allwordsTokenized1)
        r1 += 1
    elif i[13] == "2":
        bagOfReviews2.append(i[14])
        businessId2.append(i[0])
        businessName2.append(i[2])
        userId2.append(i[12])
        state2.append(i[5])
        allwordsStemmed2 = textPrepStem(i[14])
        totalvocabStemmed2.extend(allwordsStemmed2)
        allwordsTokenized2 = textPrep(i[14])
        totalvocabTokenized2.extend(allwordsTokenized2)
        r2 += 1
    elif i[13] == "3":
        bagOfReviews3.append(i[14])
        businessId3.append(i[0])
        businessName3.append(i[2])
        userId3.append(i[12])
        state3.append(i[5])
        allwordsStemmed3 = textPrepStem(i[14])
        totalvocabStemmed3.extend(allwordsStemmed3)
        allwordsTokenized3 = textPrep(i[14])
        totalvocabTokenized3.extend(allwordsTokenized3)
        r3 += 1
    elif i[13] == "4":
        bagOfReviews4.append(i[14])
        businessId4.append(i[0])
        businessName4.append(i[2])
        userId4.append(i[12])
        state4.append(i[5])
        allwordsStemmed4 = textPrepStem(i[14])
        totalvocabStemmed4.extend(allwordsStemmed4)
        allwordsTokenized4 = textPrep(i[14])
        totalvocabTokenized4.extend(allwordsTokenized4)
        r4 += 1
    elif i[13] == "5":
        bagOfReviews5.append(i[14])
        businessId5.append(i[0])
        businessName5.append(i[2])
        userId5.append(i[12])
        state5.append(i[5])
        allwordsStemmed5 = textPrepStem(i[14])
        totalvocabStemmed5.extend(allwordsStemmed5)
        allwordsTokenized5 = textPrep(i[14])
        totalvocabTokenized5.extend(allwordsTokenized5)
        r5 += 1
 
print ("1 star rating count %d " % r1)
print ("2 star rating count %d " % r2)
print ("3 star rating count %d " % r3)
print ("4 star rating count %d " % r4)
print ("5 star rating count %d " % r5)
 
for i in range(0,len(businessId1)):
    ranks1.append(i)
for i in range(0,len(businessId2)):
    ranks2.append(i)
for i in range(0,len(businessId3)):
    ranks3.append(i)
for i in range(0,len(businessId4)):
    ranks4.append(i)
for i in range(0,len(businessId5)):
    ranks5.append(i)
 
vocabFrame1 = pd.DataFrame({'words': totalvocabTokenized1}, index = totalvocabStemmed1)
vocabFrame2 = pd.DataFrame({'words': totalvocabTokenized2}, index = totalvocabStemmed2)
vocabFrame3 = pd.DataFrame({'words': totalvocabTokenized3}, index = totalvocabStemmed3)
vocabFrame4 = pd.DataFrame({'words': totalvocabTokenized4}, index = totalvocabStemmed4)
vocabFrame5 = pd.DataFrame({'words': totalvocabTokenized5}, index = totalvocabStemmed5)
 
tfidfVectorizer = TfidfVectorizer(max_df=0.8, max_features=20000000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=textPrepStem, ngram_range=(1,3))
 
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
 
km1 = KMeans(n_clusters=numClusters)
km2 = KMeans(n_clusters=numClusters)
km3 = KMeans(n_clusters=numClusters)
km4 = KMeans(n_clusters=numClusters)
km5 = KMeans(n_clusters=numClusters)
 
km1.fit(tfidfMatrix1)
km2.fit(tfidfMatrix2)
km3.fit(tfidfMatrix3)
km4.fit(tfidfMatrix4)
km5.fit(tfidfMatrix5)
 
# clustersOne = km1.labels_.tolist()
# clustersTwo = km2.labels_.tolist()
# clustersThree = km3.labels_.tolist()
# clustersFour = km4.labels_.tolist()
# clustersFive = km5.labels_.tolist()
 
joblib.dump(km1,  'doc_cluster.pkl')
# km1 = joblib.load('doc_cluster.pkl')
joblib.dump(km2,  'doc_cluster.pkl')
# km2 = joblib.load('doc_cluster.pkl')
joblib.dump(km3,  'doc_cluster.pkl')
# km3 = joblib.load('doc_cluster.pkl')
joblib.dump(km4,  'doc_cluster.pkl')
# km4 = joblib.load('doc_cluster.pkl')
joblib.dump(km5,  'doc_cluster.pkl')
# km5 = joblib.load('doc_cluster.pkl')
 
clustersOne = km1.labels_.tolist()
clustersTwo = km2.labels_.tolist()
clustersThree = km3.labels_.tolist()
clustersFour = km4.labels_.tolist()
clustersFive = km5.labels_.tolist()
 
# print clustersOne
# print clustersTwo
# print clustersThree
# print clustersFour
# print clustersFive
 
 
oneStarReviews = { 'businessId': businessId1, 'rank': ranks1, 'reviews': bagOfReviews1, 'cluster': clustersOne, 'location': state1, 'userId': userId1, 'businessName': businessName1}
twoStarReviews = { 'businessId': businessId2, 'rank': ranks2, 'reviews': bagOfReviews1, 'cluster': clustersTwo, 'location': state2, 'userId': userId2, 'businessName': businessName2}
threeStarReviews = { 'businessId': businessId3, 'rank': ranks3, 'reviews': bagOfReviews1, 'cluster': clustersThree, 'location': state3, 'userId': userId3, 'businessName': businessName3}
fourStarReviews = { 'businessId': businessId4, 'rank': ranks4, 'reviews': bagOfReviews1, 'cluster': clustersFour, 'location': state4, 'userId': userId4, 'businessName': businessName4}
fiveStarReviews = { 'businessId': businessId5, 'rank': ranks5, 'reviews': bagOfReviews1, 'cluster': clustersFive, 'location': state5, 'userId': userId5, 'businessName': businessName5}
 
reviewFrame1 = pd.DataFrame(oneStarReviews, index = [clustersOne] , columns = ['businessId', 'rank', 'cluster', 'state', 'userId'])
reviewFrame2 = pd.DataFrame(twoStarReviews, index = [clustersTwo] , columns = ['businessId', 'rank', 'cluster', 'state', 'userId'])
reviewFrame3 = pd.DataFrame(threeStarReviews, index = [clustersThree] , columns = ['businessId', 'rank', 'cluster', 'state', 'userId'])
reviewFrame4 = pd.DataFrame(fourStarReviews, index = [clustersFour] , columns = ['businessId', 'rank', 'cluster', 'state', 'userId'])
reviewFrame5 = pd.DataFrame(fiveStarReviews, index = [clustersFive] , columns = ['businessId', 'rank', 'cluster', 'state', 'userId'])
 
reviewFrame1['cluster'].value_counts()
reviewFrame2['cluster'].value_counts()
reviewFrame3['cluster'].value_counts()
reviewFrame4['cluster'].value_counts()
reviewFrame5['cluster'].value_counts()
 
# Not required for now... since we don't have any kind of ranks for the reviews
# groupOne = reviewFrame1['businessId'].groupby(reviewFrame1['cluster'])
# groupTwo = reviewFrame2['businessId'].groupby(reviewFrame2['cluster'])
# groupThree = reviewFrame3['businessId'].groupby(reviewFrame3['cluster'])
# groupFour = reviewFrame4['businessId'].groupby(reviewFrame4['cluster'])
# groupFive = reviewFrame5['businessId'].groupby(reviewFrame5['cluster'])
#
# groupOne.mean()
# groupTwo.mean()
# groupThree.mean()
# groupFour.mean()
# groupFive.mean()
 
 
# Cluster 1
print("Top terms for clusterOne:")
print()
orderCentroids = km1.cluster_centers_.argsort()[:, ::-1]
print (len(orderCentroids))
for i in range(numClusters):
    print("Cluster %d words:" % i, end='')
    x1 = len(orderCentroids[i,:]) - 1
    print ("x1: ", x1)
    if x1 > 6:
        x1 = 6
    print ("x1: ", x1)
    # for ind in orderCentroids[i, :6]:
    #     print(' %s' % vocabFrame1.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d BusinessId:" % i, end='')
    for feature in reviewFrame1.ix[i]['businessId'].values.tolist():
        print(' %s, ' % feature)
    print()
    print()
 
# Cluster 2
print("Top terms for clusterTwo:")
print()
orderCentroids = km2.cluster_centers_.argsort()[:, ::-1]
if len(orderCentroids) < 6:
    x2 = len(orderCentroids) + 1
for i in range(numClusters):
    print("Cluster %d words:" % i, end='')
    # for ind in orderCentroids[i, :x2]:
    #     print(' %s' % vocabFrame2.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d BusinessId:" % i, end='')
    for feature in reviewFrame2.ix[i]['businessId'].values.tolist():
        print(' %s, ' % feature)
    print()
    print()
 
# Cluster 3
print("Top terms for clusterThree:")
print()
orderCentroids = km3.cluster_centers_.argsort()[:, ::-1]
if len(orderCentroids) < 6:
    x3 = len(orderCentroids) + 1
for i in range(numClusters):
    print("Cluster %d words:" % i, end='')
    print (len(orderCentroids))
    # for ind in orderCentroids[i, :x3]:
    #     print(' %s' % vocabFrame3.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d BusinessId:" % i, end='')
    for feature in reviewFrame3.ix[i]['businessId'].values.tolist():
        print(' %s, ' % feature)
    print()
    print()
 
# Cluster 4
print("Top terms for clusterFour:")
print()
orderCentroids = km4.cluster_centers_.argsort()[:, ::-1]
if len(orderCentroids) < 6:
    x4 = len(orderCentroids) + 1
for i in range(numClusters):
    print("Cluster %d words:" % i, end='')
    print (len(orderCentroids))
    # for ind in orderCentroids[i, :x4]:
    #     print(' %s' % vocabFrame4.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d BusinessId:" % i, end='')
    for feature in reviewFrame4.ix[i]['businessId'].values.tolist():
        print(' %s, ' % feature)
    print()
    print()
 
# Cluster 5
print("Top terms for clusterFive:")
print()
orderCentroids = km5.cluster_centers_.argsort()[:, ::-1]
if len(orderCentroids) < 6:
    x5 = len(orderCentroids) + 1
for i in range(numClusters):
    print("Cluster %d words:" % i, end='')
    print (len(orderCentroids))
    # for ind in orderCentroids[i, :]:
    #     print(' %s' % vocabFrame5.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d BusinessId:" % i, end='')
    for feature in reviewFrame5.ix[i]['businessId'].values.tolist():
        print(' %s, ' % feature)
    print()
    print()
 
reviewFrame1['bizId'] = reviewFrame1['businessId']
reviewFrame2['bizId'] = reviewFrame2['businessId']
reviewFrame3['bizId'] = reviewFrame3['businessId']
reviewFrame4['bizId'] = reviewFrame4['businessId']
reviewFrame5['bizId'] = reviewFrame5['businessId']
 
print(reviewFrame1[['bizId']].loc[reviewFrame1['cluster'] == 1].to_html(index=False))
print(reviewFrame2[['bizId']].loc[reviewFrame2['cluster'] == 1].to_html(index=False))
print(reviewFrame3[['bizId']].loc[reviewFrame3['cluster'] == 1].to_html(index=False))
print(reviewFrame4[['bizId']].loc[reviewFrame4['cluster'] == 1].to_html(index=False))
print(reviewFrame5[['bizId']].loc[reviewFrame5['cluster'] == 1].to_html(index=False))
 
 
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
 
pos1 = mds.fit_transform(dist1)  # shape (n_components, n_samples)
pos2 = mds.fit_transform(dist2)  # shape (n_components, n_samples)
pos3 = mds.fit_transform(dist3)  # shape (n_components, n_samples)
pos4 = mds.fit_transform(dist4)  # shape (n_components, n_samples)
pos5 = mds.fit_transform(dist5)  # shape (n_components, n_samples)
 
xs1, ys1 = pos1[:, 0], pos1[:, 1]
xs2, ys2 = pos2[:, 0], pos2[:, 1]
xs3, ys3 = pos3[:, 0], pos3[:, 1]
xs4, ys4 = pos4[:, 0], pos4[:, 1]
xs5, ys5 = pos5[:, 0], pos5[:, 1]
 
# "In [49]" Visualizing document clusters

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# Need to add in the actual words that popup
#set up cluster names using a dict
cluster_names = {0: 'Family, home, war',
                 1: 'Police, killed, murders',
                 2: 'Father, New York, brothers',
                 3: 'Dance, singing, love',
                 4: 'Killed, soldiers, captain'}

# matplotlib inline
#create data frame that has the result of the MDS plus the cluster numbers and titles
df1 = pd.DataFrame(dict(x=xs1, y=ys1, label=clustersOne, title=businessName1))
df2 = pd.DataFrame(dict(x=xs2, y=ys2, label=clustersTwo, title=businessName2))
df3 = pd.DataFrame(dict(x=xs3, y=ys3, label=clustersThree, title=businessName3))
df4 = pd.DataFrame(dict(x=xs4, y=ys4, label=clustersFour, title=businessName4))
df5 = pd.DataFrame(dict(x=xs5, y=ys5, label=clustersFive, title=businessName5))

#group by cluster
groups1 = df1.groupby('label')
groups2 = df2.groupby('label')
groups3 = df3.groupby('label')
groups4 = df4.groupby('label')
groups5 = df5.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label

# 1 star rating cluster
for name, group in groups1:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
ax.legend(numpoints=1)  #show legend with only 1 point
#add label in x,y position with the label as the film title
for i in range(len(df1)):
    ax.text(df1.ix[i]['x'], df1.ix[i]['y'], df1.ix[i]['title'], size=8)
# plt.show() #show the plot
#uncomment the below to save the plot if need be
plt.savefig('oneStarCluster.png', dpi=200)

# 2 star rating cluster
for name, group in groups2:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
ax.legend(numpoints=1)  #show legend with only 1 point
#add label in x,y position with the label as the film title
for i in range(len(df2)):
    ax.text(df2.ix[i]['x'], df2.ix[i]['y'], df2.ix[i]['title'], size=8)
# plt.show() #show the plot
#uncomment the below to save the plot if need be
plt.savefig('twoStarCluster.png', dpi=200)

# 3 star rating cluster
for name, group in groups3:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
ax.legend(numpoints=1)  #show legend with only 1 point
#add label in x,y position with the label as the film title
for i in range(len(df3)):
    ax.text(df3.ix[i]['x'], df3.ix[i]['y'], df3.ix[i]['title'], size=8)
# plt.show() #show the plot
#uncomment the below to save the plot if need be
plt.savefig('threeStarCluster.png', dpi=200)

# 4 star rating cluster
for name, group in groups4:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
ax.legend(numpoints=1)  #show legend with only 1 point
#add label in x,y position with the label as the film title
for i in range(len(df4)):
    ax.text(df4.ix[i]['x'], df4.ix[i]['y'], df4.ix[i]['title'], size=8)
# plt.show() #show the plot
#uncomment the below to save the plot if need be
plt.savefig('fourStarCluster.png', dpi=200)

# 5 star rating cluster
for name, group in groups5:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
ax.legend(numpoints=1)  #show legend with only 1 point
#add label in x,y position with the label as the film title
for i in range(len(df5)):
    ax.text(df5.ix[i]['x'], df5.ix[i]['y'], df5.ix[i]['title'], size=8)
# plt.show() #show the plot
#uncomment the below to save the plot if need be
plt.savefig('fiveStarCluster.png', dpi=200)

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
print ("Hakuna MaTaTaAaAaaaaA!!")

#create data frame that has the result of the MDS plus the cluster numbers and titles
df1 = pd.DataFrame(dict(x=xs1, y=ys1, label=clustersOne, title=businessName1))
df2 = pd.DataFrame(dict(x=xs2, y=ys2, label=clustersTwo, title=businessName2))
df3 = pd.DataFrame(dict(x=xs3, y=ys3, label=clustersThree, title=businessName3))
df4 = pd.DataFrame(dict(x=xs4, y=ys4, label=clustersFour, title=businessName4))
df5 = pd.DataFrame(dict(x=xs5, y=ys5, label=clustersFive, title=businessName5))

#group by cluster
groupsOne = df1.groupby('label')
groupsTwo = df2.groupby('label')
groupsThree = df3.groupby('label')
groupsFour = df4.groupby('label')
groupsFive = df5.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }
"""

# Plot 1
fig1, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groupsOne:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig1, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# uncomment the below to export to html
html1 = mpld3.fig_to_html(fig1)
# print(html)
htmlFile= open("plot1.html","w")
htmlFile.write(html1)
htmlFile.close()

# Plot 2
fig2, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groupsTwo:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig2, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# uncomment the below to export to html
html2 = mpld3.fig_to_html(fig2)
# print(html)
htmlFile= open("plot2.html","w")
htmlFile.write(html2)
htmlFile.close()

# Plot 3
fig3, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groupsThree:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig3, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# uncomment the below to export to html
html3 = mpld3.fig_to_html(fig3)
# print(html)
htmlFile= open("plot3.html","w")
htmlFile.write(html3)
htmlFile.close()

# Plot 4
fig4, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groupsFour:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig4, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# uncomment the below to export to html
html4 = mpld3.fig_to_html(fig4)
# print(html)
htmlFile= open("plot4.html","w")
htmlFile.write(html4)
htmlFile.close()


# Plot 5
fig5, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groupsFive:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig5, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

# uncomment the below to export to html
html5 = mpld3.fig_to_html(fig5)
# print(html)
htmlFile= open("plot5.html","w")
htmlFile.write(html5)
htmlFile.close()



# Dendogram 1

linkageMatrix1 = ward(dist1) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkageMatrix1, orientation="right", labels=businessName1);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('wardClustersOne.png', dpi=200) #save figure as ward_clusters

# Dendogram 2

linkageMatrix2 = ward(dist2) #define the linkage_matrix using ward clustering pre-computed distances

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkageMatrix2, orientation="right", labels=businessName2);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('wardClustersTwo.png', dpi=200) #save figure as ward_clusters

# Dendogram 3

linkageMatrix3 = ward(dist3) #define the linkage_matrix using ward clustering pre-computed distances

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkageMatrix3, orientation="right", labels=businessName3);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('wardClustersThree.png', dpi=200) #save figure as ward_clusters

# Dendogram 4

linkageMatrix4 = ward(dist4) #define the linkage_matrix using ward clustering pre-computed distances

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkageMatrix4, orientation="right", labels=businessName4);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('wardClustersFour.png', dpi=200) #save figure as ward_clusters

# Dendogram 5

linkageMatrix5 = ward(dist5) #define the linkage_matrix using ward clustering pre-computed distances

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkageMatrix5, orientation="right", labels=businessName5);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('wardClustersFive.png', dpi=200) #save figure as ward_clusters

#Latent Dirichlet Allocation implementation with Gensim

# LDA 1 Star
preprocess1 = [stripProppersPOS(doc) for doc in bagOfReviews1]
tokenizedText1 = [textPrepStem(text) for text in preprocess1]
texts1 = [[word for word in text if word not in stopwords] for text in tokenizedText1]
print (len(texts1))
dictionary1 = corpora.Dictionary(texts1)
dictionary1.filter_extremes(no_below=1, no_above=0.8)
corpus1 = [dictionary1.doc2bow(text) for text in texts1]
len(corpus1)
lda1 = models.LdaModel(corpus1, num_topics=5, id2word=dictionary1, update_every=5, chunksize=10000, passes=100)
print(lda1[corpus1])
topics1 = lda1.print_topics(5, num_words=20)
topicMatrix1 = lda1.show_topics(formatted=False, num_words=20)
topicMatrix1 = np.array(topicMatrix1)
topicMatrix1.shape
topicWords1 = topicMatrix1[:,:,1]
for i in topicWords1:
    print([str(word) for word in i])
    print()
    
# LDA 1 Star
preprocess2 = [stripProppersPOS(doc) for doc in bagOfReviews2]
tokenizedText2 = [textPrepStem(text) for text in preprocess2]
texts2 = [[word for word in text if word not in stopwords] for text in tokenizedText2]
print (len(texts2))
dictionary2 = corpora.Dictionary(texts2)
dictionary2.filter_extremes(no_below=1, no_above=0.8)
corpus2 = [dictionary2.doc2bow(text) for text in texts2]
len(corpus2)
lda2 = models.LdaModel(corpus2, num_topics=5, id2word=dictionary2, update_every=5, chunksize=10000, passes=100)
print(lda2[corpus2])
topics2 = lda2.print_topics(5, num_words=20)
topicMatrix2 = lda2.show_topics(formatted=False, num_words=20)
topicMatrix2 = np.array(topicMatrix2)
topicMatrix2.shape
topicWords2 = topicMatrix2[:,:,1]
for i in topicWords2:
    print([str(word) for word in i])
    print()
    
# LDA 3 Star
preprocess3 = [stripProppersPOS(doc) for doc in bagOfReviews3]
tokenizedText3 = [textPrepStem(text) for text in preprocess3]
texts3 = [[word for word in text if word not in stopwords] for text in tokenizedText3]
print (len(texts3))
dictionary3 = corpora.Dictionary(texts3)
dictionary3.filter_extremes(no_below=1, no_above=0.8)
corpus3 = [dictionary3.doc2bow(text) for text in texts3]
len(corpus3)
lda3 = models.LdaModel(corpus3, num_topics=5, id2word=dictionary3, update_every=5, chunksize=10000, passes=100)
print(lda3[corpus3])
topics3 = lda3.print_topics(5, num_words=20)
topicMatrix3 = lda3.show_topics(formatted=False, num_words=20)
topicMatrix3 = np.array(topicMatrix3)
topicMatrix3.shape
topicWords3 = topicMatrix3[:,:,1]
for i in topicWords3:
    print([str(word) for word in i])
    print()
    
# LDA 4 Star
preprocess4 = [stripProppersPOS(doc) for doc in bagOfReviews4]
tokenizedText4 = [textPrepStem(text) for text in preprocess4]
texts4 = [[word for word in text if word not in stopwords] for text in tokenizedText4]
print (len(texts4))
dictionary4 = corpora.Dictionary(texts4)
dictionary4.filter_extremes(no_below=1, no_above=0.8)
corpus4 = [dictionary4.doc2bow(text) for text in texts4]
len(corpus4)
lda4 = models.LdaModel(corpus4, num_topics=5, id2word=dictionary4, update_every=5, chunksize=10000, passes=100)
print(lda4[corpus4])
topics4 = lda4.print_topics(5, num_words=20)
topicMatrix4 = lda4.show_topics(formatted=False, num_words=20)
topicMatrix4 = np.array(topicMatrix4)
topicMatrix4.shape
topicWords4 = topicMatrix4[:,:,1]
for i in topicWords4:
    print([str(word) for word in i])
    print()
    
# LDA 5 Star
preprocess5 = [stripProppersPOS(doc) for doc in bagOfReviews5]
tokenizedText5 = [textPrepStem(text) for text in preprocess5]
texts5 = [[word for word in text if word not in stopwords] for text in tokenizedText5]
print (len(texts5))
dictionary5 = corpora.Dictionary(texts5)
dictionary5.filter_extremes(no_below=1, no_above=0.8)
corpus5 = [dictionary5.doc2bow(text) for text in texts5]
len(corpus5)
lda5 = models.LdaModel(corpus5, num_topics=5, id2word=dictionary5, update_every=5, chunksize=10000, passes=100)
print(lda1[corpus5])
topics5 = lda5.print_topics(5, num_words=20)
topicMatrix5 = lda5.show_topics(formatted=False, num_words=20)
topicMatrix5 = np.array(topicMatrix5)
topicMatrix5.shape
topicWords5 = topicMatrix5[:,:,1]
for i in topicWords5:
    print([str(word) for word in i])
    print()
