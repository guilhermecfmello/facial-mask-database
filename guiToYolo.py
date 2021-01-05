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
# Time for progress bar
import time



sourcePath = "guilhermecfmello/not-cropped/"
annPath = sourcePath + "annotations/"
imgPath = sourcePath + "images/"
destinyPath = "guilhermecfmello/yolo-format-90-10/"

def annotationConvert(ann, classMap):
    an = ET.parse(annPath+ann)
    root = an.getroot()
    newAnns = bndBoxToYolo(root, classMap)
    imgName = root.find('filename').text
    txtName = imgName[:imgName.find('.')] + '.txt'
    copyImage(imgPath, imgName, destinyPath, imgName)
    writeYoloAnn(destinyPath, txtName, newAnns)

# Write the yoloAnn on disk
def writeYoloAnn(path, fileName, anns):
    f = open(path+fileName, 'w')
    for ann in anns:
        f.write(ann)
    f.close()

# Copy the image from source to destiny path
def copyImage(sourcePath, imgName, destinyPath, destinyName):
    cp(sourcePath+imgName, destinyPath+destinyName)
    return True

# Bounding box PASCAL VOC convertion format to Yolo annotation format
def bndBoxToYolo(root, classMap):
    size = root.find('size')
    imgName = root.find('filename').text
    imgWidth = int(size.find('width').text)
    imgHeight = int(size.find('height').text)
    objs = root.findall('object')
    newAnns = []
    for obj in objs:
        objC = obj.find('name').text
        newClass = classMap[objC]
        xMin, yMin, xMax, yMax = getObjBndBox(obj)
        w = xMax - xMin
        h = yMax - yMin
        xCenter = (w/2)+xMin
        yCenter = (h/2)+yMin

        # Normalizing values
        xCenter /= imgWidth
        yCenter /= imgHeight
        w /= imgWidth
        h /= imgHeight

        yoloAnn = str(newClass) + ' ' 
        yoloAnn += str(xCenter) + ' '
        yoloAnn += str(yCenter) + ' '
        yoloAnn += str(w) + ' '
        yoloAnn += str(h) + '\n'
        newAnns.append(yoloAnn)
    return newAnns


def getObjBndBox(obj):
    bnd = obj.find('bndbox')
    xMin = int(bnd.find('xmin').text)
    yMin = int(bnd.find('ymin').text)
    xMax = int(bnd.find('xmax').text)
    yMax = int(bnd.find('ymax').text)
    return xMin, yMin, xMax, yMax

classesMap = {
    'without_mask': 0,
    'with_mask': 1,
    'mask_weared_incorrect': 2
}

annsList = [f for f in listdir(annPath) if isfile(join(annPath, f))]

toolbar_width = 60
total = len(annsList)
# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
i = 0
for annName in annsList:
    annotationConvert(annName, classesMap)
    # update the bar
    if((i % 72) == 0):
        sys.stdout.write("-")
        sys.stdout.flush()
    i += 1

sys.stdout.write("]\n") # this ends the progress bar

print('number of annotations: ' + str(len(annsList)))
# print(annsList)


    