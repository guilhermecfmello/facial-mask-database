# xml import
import xml.etree.ElementTree as ET
import xml.dom.minidom
# json import
import json
# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch
# Image
from PIL import Image


# Verifiy if json file has some classname defined in maskFilter
def hasValidClass(obj, maskFilter):
    for ann in obj['Annotations']:
        if annIsValid(ann, maskFilter):
            return True
    return False


def annIsValid(ann, maskfilter):
    if ann['classname'] in maskFilter:
        return True
    else: 
        return False

# Get Image name of jsonFile
def getImgName(obj):
    return obj['FileName']

# create an Array with valid annotations defined in maskFilter
def getValidAnnsXml(obj, maskFilter):
    annArray = []
    for ann in obj['Annotations']:
        if annIsValid(ann, maskFilter):
            newAnn = ET.Element('object')
            ET.SubElement(newAnn, 'name').text = maskFilter[ann['classname']]
            ET.SubElement(newAnn, 'pose').text = 'Unspecified'
            ET.SubElement(newAnn, 'truncated').text = '0'
            ET.SubElement(newAnn, 'occluded').text = '0'
            ET.SubElement(newAnn, 'difficult').text = '0'
            bndbox = ET.SubElement(newAnn, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = '777'
            ET.SubElement(bndbox, 'ymin').text = '888'
            ET.SubElement(bndbox, 'xmax').text = '999'
            ET.SubElement(bndbox, 'ymax').text = '000'
            annArray = annArray + [newAnn]
    return annArray

def createAnn(fileName, width, height, anns):
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text = 'images'
    ET.SubElement(ann, 'filename').text = fileName

    size = ET.SubElement(ann, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(3)

    ET.SubElement(ann, 'segmented').text = str(0)

    for a in anns:
        newObj = ET.SubElement(ann, 'object')
        newObj.extend(a) 
    return ann


annsPath = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/annotations/'
imgPath = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/images/'

# className to filter images
maskFilter = {
    # 'other':'with_mask', # Needs pos treatment
    # 'scarf_bandana':'with_mask', # Needs pos treatment
    # 'face_other_covering':'with_mask', # Needs pos treatment
    'face_with_mask':'with_mask',
    'face_with_mask_incorrect':'mask_weared_incorrect',
    'face_no_mask':'without_mask',
    'mask_surgical':'with_mask',
    'mask_colorful' : 'with_mask' 
}

# fullPath = 
annsList = [f for f in listdir(annsPath) if isfile(join(annsPath, f))]

# List of files inserted on new database 
filesList = {}

cont = 0
# print(annsList[:2])
for ann in annsList:
    jsonFile = open(annsPath+ann)
    jsonObj = json.load(jsonFile)
    

    anns = getValidAnnsXml(jsonObj, maskFilter)
    if len(anns) > 1:
        cont = cont + 1
        # getting images informations
        imgName = getImgName(jsonObj)
        im = Image.open(imgPath+imgName)
        width, height = im.size

        # print(jsonObj)
        # exit()
        # Creating new Annotation file 
        newXml = createAnn(imgName, width, height, anns)


        tree = ET.ElementTree(newXml)
        tree.write('xmltest.xml')


        exit()

print('valid class amount:' + str(cont))
# print(len(annsList))
