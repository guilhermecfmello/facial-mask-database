# xml import
import xml.etree.ElementTree as ET
# import requests
# json import
import json
# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch
# copy file
import shutil


# className to filter images
maskFilter = ['other']

path = 'guilhermecfmello/cropped/test/'

examplePath = 'guilhermecfmello/cropped/test/examples/'
anns = 'annotations/'
imgs = 'images/'

fullPath = path + anns
# print(listdir(fullPath))
# exit()
annsList = [f for f in listdir(fullPath) if isfile(join(fullPath, f))]
annsList = [x for x in annsList if x[len(x)-3:] != 'xml']
# print(annsList)
# exit()
classes = {}
for ann in annsList:
    # print(fullPath+ann)
    # exit()
    jsonFile = open(fullPath+ann)
    print(jsonFile)
    exit()
    data = json.load(jsonFile)
    for maskType in data['Annotations']:
        m = maskType['classname']
        if m in classes:
            classes[m] = classes[m] + 1
        else:
            classes[m] = 1
        if(classes[m] <= 3): 
            shutil.copyfile(path+imgs+data['FileName'], examplePath+m+"example"+str(classes[m]))
            print('printing ' +path+imgs+data['FileName'])
