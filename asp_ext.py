__author__ = 'maruthi'
 
import re
import sys
import csv
from nltk import pos_tag
# from collections import Counter
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import RegexpTokenizer #Regular Expression Based Tokenizer
# from nltk.corpus import stopwords #Stopwords List
import xlsxwriter as xlS
 
 
def ext1(fileName,textColumn):
    row0 = 0
    col0 = 0
    x = 0
    wordcount = {}
 
    dataOpen = open(fileName, 'r')
 
    pickLoc = fileName.find(".")
    newFile = fileName[:pickLoc]
    workbook = xlS.Workbook(newFile + "_extraction.xlsx")
    worksheet0 = workbook.add_worksheet('Nouns1')
    worksheet0.write(row0, col0, "Nouns")
    worksheet0.write(row0, col0 + 1, "Count")
 
 
    # d1 = csv.reader(dataOpen,delimiter=',',quotechar='"')
 
 
 
    for revieW in dataOpen:
        if x > 0:
            revieW = revieW.split(",")
            tagged_sent = pos_tag(revieW[textColumn].split())
            aspectS = [word.lower() for word,pos in tagged_sent if pos == 'NNP' or pos == 'JJ' or pos == 'JJS' or pos == 'JJR']
            for word in aspectS:
                word = re.sub(r'[^\w\s]','',word)
                word = re.sub(r'[0-9]','',word)
                if word not in wordcount:
                    if word != '':
                        wordcount[word] = 1
                    else:
                        continue
                else:
                    wordcount[word] += 1
        else:
            x += 1
            continue
    for k,v in wordcount.items():
        worksheet0.write(row0 + 1, col0, k)
        worksheet0.write(row0 + 1, col0 + 1, v)
        row0 += 1
 
    # print wordcount
 
    workbook.close()
 
if __name__ == '__main__':
    try:
        filE = sys.argv[1]
        loC = int(sys.argv[2])
        ext1(filE,loC)
        print "Hakuna mAtAtA!!"
    except IndexError:
        print 'Usage: python asp_ext.py <file name> <text - column index number>'
