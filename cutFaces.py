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
# Cut image
from PIL import Image

def getCoordinates(object):
  bndbox = obj.find('bndbox')
  xMin = int(bndbox.find('xmin').text)
  xMax = int(bndbox.find('xmax').text)
  yMin = int(bndbox.find('ymin').text)
  yMax = int(bndbox.find('ymax').text)
  # print(object.text)
  # exit()

  return xMin, yMin, xMax, yMax


validationAnn = 'guilhermecfmello/validation/annotations/'
validationImg = 'guilhermecfmello/validation/images/'
croppedAnn = 'guilhermecfmello/cropped/validation/annotations/'
validationImg = 'guilhermecfmello/cropped/validation/images/'


annsList = [f for f in listdir(validationAnn) if isfile(join(validationAnn, f))]

for ann in annsList:
  if ann == '5619w.xml':
    with open(validationAnn+ann,'r') as file:
      tree = ET.fromstring(file.read())
      imgName = tree.find('filename').text
      img = Image.open(validationImg+imgName)
      
      # print(tree.findall('object')[1].text)
      # exit()
      for obj in tree.iter('object'):
        # print(getCoordinates(obj))
        # print(obj.find('bndbox').find('xmin').text)
        # exit(0)
        xMin, Ymin, xMax, yMax = getCoordinates(obj)
        img2 = img.crop((xMin, Ymin, xMax, yMax))
        img2.show()
        # exit()
      
      exit(0)
      # objs = tree.findall('object')
      # print(len(objs))


