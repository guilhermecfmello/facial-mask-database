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
# Args
import sys



sourcePath = "guilhermecfmello/not-cropped/"
annPath = sourcePath + "annotations/"
imgPath = sourcePath + "images/"
destinyPath = "guilhermecfmello/yolo-format/"

def annotationConvert(ann, classMap):
    an = ET.parse(annPath+ann)
    newAnns = bndBoxToYolo(an, classMap)


# Bounding box PASCAL VOC convertion format to Yolo annotation format
def bndBoxToYolo(an, classMap):
    root = an.getroot()
    size = root.findall('size')[0]
    imgName = root.findall('filename')[0].text
    imgWidth = size[0].text
    imgHeight = size[1].text
    objs = root.findall('object')
    newAnns = []
    for obj in objs:
        newClass = classMap[obj[0].text]


classesMap = {
    'without_mask': 0,
    'with_mask': 1,
    'mask_weared_incorect': 2
}

annsList = [f for f in listdir(annPath) if isfile(join(annPath, f))]

i = 0
for annName in annsList:
    annotationConvert(annName, classesMap)
    if i == 1:
        exit()
    i += 1

print('number of annotations: ' + str(len(annsList)))
print(annsList)


    