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


sourcePath = 'guilhermecfmello/yolo-format-reduced/reduced/validation/'
destinyPath = 'guilhermecfmello/yolo-format-reduced/reduced/'
prefix = 'data/validation/'

imgsList = []
filesList = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]
for f in filesList:
    dot = f.find('.')
    if f[dot:] != '.txt':
        imgsList.append(f)

f = open(destinyPath+'validation.txt', 'w')
for img in imgsList:
    f.write(prefix+img+'\n')
f.close()

