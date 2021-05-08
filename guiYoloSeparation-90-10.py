# xml import
import xml.etree.ElementTree as ET
from xml.dom import minidom
# json import
import json
# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch
# Image
from PIL import Image
# Copy file
from shutil import copy as cp
from shutil import move as mv
from shutil import rmtree as rmTree
# Args
import sys
# Time for progress bar
import time


sourcePath = 'guilhermecfmello/yolo-format-reduced/reduced/'
trainingPath = 'guilhermecfmello/yolo-format-reduced/reduced/training/'
validationPath = 'guilhermecfmello/yolo-format-reduced/reduced/validation/'

filesList = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]

# Number of images/annotations for training, validation
nLarxel = [95, 9]
nWoobot = [166, 19]

nLarxelBk = [95, 9]
nWoobotBk = [166, 19]

# Getting only image files
imgsList = []
for f in filesList:
    dot = f.find('.')
    if f[dot:] != '.txt':
        imgsList.append(f)

annsTraining = []
imgsTraining = []

annsValidation = []
imgsValidation = []

for img in imgsList:
    dot = img.find('.')
    annName = img[:dot] + '.txt'

    # Is a larxel
    if img[0] == 'm':
        print(nLarxel)
        # training
        if nLarxel[0] > 0:
            cp(sourcePath+annName, trainingPath+annName)
            cp(sourcePath+img, trainingPath+img)
            annsTraining.append(trainingPath+annName)
            imgsTraining.append(trainingPath+img)
            nLarxel[0] -= 1
        # validation
        elif nLarxel[1] > 0:
            cp(sourcePath+annName, validationPath+annName)
            cp(sourcePath+img, validationPath+img)
            annsValidation.append(validationPath+annName)
            imgsValidation.append(validationPath+img)
            nLarxel[1] -= 1
    #  Is an woobot
    else:
        # training
        if nWoobot[0] > 0:
            cp(sourcePath+annName, trainingPath+annName)
            cp(sourcePath+img, trainingPath+img)
            annsTraining.append(trainingPath+annName)
            imgsTraining.append(trainingPath+img)
            nWoobot[0] -= 1
        # validation
        elif nWoobot[1] > 0:
            cp(sourcePath+annName, validationPath+annName)
            cp(sourcePath+img, validationPath+img)
            annsValidation.append(validationPath+annName)
            imgsValidation.append(validationPath+img)
            nWoobot[1] -= 1

# Files verification for training
filesList = [f for f in listdir(trainingPath) if isfile(join(trainingPath, f))]
size = int(len(filesList)/2)
print("Number of files that should be in training: " + str((nLarxelBk[0] + nWoobotBk[0])))
print("Number of files that is there: " + str(size))
if size != (nLarxelBk[0] + nWoobotBk[0]):
    print("Error on training verification, removing files...")
    for ann in annsTraining:
        os.remove(ann)
    for img in imgsTraining:
        os.remove(img)
    print("Files removed!")
    
# Files verification for validation
filesList = [f for f in listdir(validationPath) if isfile(join(validationPath, f))]
size = int(len(filesList)/2)
print("Number of files that should be in validation: " + str((nLarxelBk[1] + nWoobotBk[1])))
print("Number of files that is there: " + str(size))
if size != (nLarxelBk[1] + nWoobotBk[1]):
    print("Error on validation verification, removing files...")
    for ann in annsValidation:
        os.remove(ann)
    for img in imgsValidation:
        os.remove(img)
    print("Files removed!")


print("Done")
print("larxel:")
print(nLarxel)
print("woobot:")
print(nWoobot)



