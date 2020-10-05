# xml import
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import tostring
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
# Cut image
from PIL import Image

def getCoordinates(object):
  bndbox = obj.find('bndbox')
  xMin = int(bndbox.find('xmin').text)
  xMax = int(bndbox.find('xmax').text)
  yMin = int(bndbox.find('ymin').text)
  yMax = int(bndbox.find('ymax').text)
  return xMin, yMin, xMax, yMax

path = 'guilhermecfmello/training/'
destinyPath = 'guilhermecfmello/cropped/training/'

sourceAnn = path + 'annotations/'
sourceImg = path + 'images/'

destinyAnn = destinyPath + 'annotations/'
destinyImg = destinyPath + 'images/'

def newXml(fileName, width, height, obj):
  newAnn = ET.Element('annotation')
  ET.SubElement(newAnn, 'folder').text = 'images'
  ET.SubElement(newAnn, 'filename').text = fileName
  size = ET.SubElement(newAnn, 'size')
  ET.SubElement(size, 'width').text = str(width)
  ET.SubElement(size, 'height').text = str(height)
  ET.SubElement(size, 'depth').text = '3'
  ET.SubElement(newAnn, 'segmented').text = '0'
  newObj = ET.SubElement(newAnn, 'object')
  newObj.extend(obj)
  return newAnn

annsList = [f for f in listdir(sourceAnn) if isfile(join(sourceAnn, f))]

for ann in annsList:
  with open(sourceAnn+ann,'r') as file:
    tree = ET.fromstring(file.read())
    imgName = tree.find('filename').text
    img = Image.open(sourceImg+imgName)
    dot = imgName.find('.')
    i = 1
    for obj in tree.iter('object'):
      baseName = imgName[:dot] + 'face-' + str(i)
      print('Creating ' + baseName + '...')
      xMin, Ymin, xMax, yMax = getCoordinates(obj)
      try:
        tinyImg = img.crop((xMin, Ymin, xMax, yMax))
        tinyImg.save(destinyImg + baseName + imgName[dot:])
        width, height = tinyImg.size
        newAnn = newXml(baseName, width, height, obj)
        xmlStr = minidom.parseString(ET.tostring(newAnn)).toprettyxml(indent='  ')
        with open(destinyAnn + baseName + '.xml', 'w') as f:
          # printing xml as string without header 'version'
          f.write(xmlStr[23:])
        i = i + 1
      except:
        print("@@@@@@@@@@@@@@@@@@@ ERROR ON CROP @@@@@@@@@@@@@@@@@@@")
        with open('errors-crop.log', 'w+') as f:
          f.write('error on: ' + baseName)
      


