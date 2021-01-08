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

sourcePath = "guilhermecfmello/dataset-1.2/"
annPath = sourcePath + "annotations/"
imgPath = sourcePath + "images/"
destinyPath = "guilhermecfmello/labelme-format/dataset/"

def annotationConvert(ann, classMap):
    an = ET.parse(annPath+ann)
    annName = ann
    root = an.getroot()
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    ann = {}

    shapes = bndBoxToShapesLabelme(root, classMap)
    # constants
    version = "4.5.6"
    flags = []

    # annotations infos
    imgName = root.find('filename').text
    # imgData = getImageData(imgPath, imgName)
    # print(imgData)
    # exit()
    ann["version"] = version
    ann["flags"] = flags
    ann["shapes"] = shapes
    ann["imagePath"] = imgName
    ann["imageHeight"] = height
    ann["imageData"] = None
    ann["imageWidth"] = width
    # each face
    fileName = imgName[:imgName.find('.')] + '.json'
    # print(json.dumps(ann))
    # exit()
    writeToLabelme(destinyPath, fileName, ann)
    copyImage(imgPath, imgName, destinyPath, imgName)

# Copy the image from source to destiny path
def copyImage(sourcePath, imgName, destinyPath, destinyName):
    cp(sourcePath+imgName, destinyPath+destinyName)
    return True

# Write the yoloAnn on disk
def writeToLabelme(path, fileName, ann):
    f = open(path+fileName, 'w')
    f.write(json.dumps(ann))
    f.close()

def getImageData(sourcePath, imgName):
    f = open(sourcePath+imgName, "r")
    data = f.read()
    f.close()
    return data

# Bounding box PASCAL VOC convertion format to Yolo annotation format
def bndBoxToShapesLabelme(root, classMap):
    shape_type = "rectangle"
    flags = {}
    group_id = None

    objs = root.findall('object')
    shapes = []
    for obj in objs:
        shape = {}
        objC = obj.find('name').text

        xMin, yMin, xMax, yMax = getObjBndBox(obj)
        points = [[xMin, yMin],[xMax, yMax]]
        label = classMap[objC]

        shape["label"] = label
        shape["points"] = points
        shape["group_id"] = group_id
        shape["shape_type"] = shape_type
        shape["flags"] = flags

        shapes.append(shape)
    return shapes


def getObjBndBox(obj):
    bnd = obj.find('bndbox')
    xMin = int(bnd.find('xmin').text)
    yMin = int(bnd.find('ymin').text)
    xMax = int(bnd.find('xmax').text)
    yMax = int(bnd.find('ymax').text)
    return xMin, yMin, xMax, yMax

classesMap = {
    'without_mask': "0",
    'with_mask': "1",
    'mask_weared_incorrect': "2"
}

annsList = [f for f in listdir(annPath) if isfile(join(annPath, f))]

# annotationConvert('1801w.xml', classesMap)


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


    