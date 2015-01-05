__author__ = 'maruthi'

import json
import re
import pandas

# print "good"
cn = 0
cn1 = 0
hdr = list()
source = open("Dataset/yelp_academic_dataset_user.json")
with open("Dataset/yelp_academic_dataset_user.json") as a:
    data = a.read()
    jdata = json.loads(data)
    # print jdata
    # print type(jdata)
    for x in jdata:
        cn = cn + 1
        # print range(len(x))
        # print x
        if cn > 7: break
        for z, y in x.items():
            # if z not in hdr:
            # hdr.append(z)
            if y == dict():
                for i, u in y.items():
                    print i, u
                    hdr.append(u)
                    print u
            else:
                hdr.append(z)
                print z
                # print " :) !!"

            if z in ['votes','compliments']:
                print z
            #
            #
            # print z
    # for y in hdr1:
    #     print y
    #     if y not in hdr:
    #         hdr.append(y)
            # print y
            # else: continue
            # print type(z)
            # print z
            cn1 = cn1 + 1
            if cn1 > 30: break
# t = list(hdr)
header = hdr
# print header
# print t
# print hdr
# print jdata

# print source