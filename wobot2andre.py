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
            ET.SubElement(bndbox, 'xmin').text = str(ann['BoundingBox'][0])
            ET.SubElement(bndbox, 'ymin').text = str(ann['BoundingBox'][1])
            ET.SubElement(bndbox, 'xmax').text = str(ann['BoundingBox'][2])
            ET.SubElement(bndbox, 'ymax').text = str(ann['BoundingBox'][3])
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


# copy all files from source directory to destination
def copyDirectory(source, destination):
    if os.path.isdir(source) and os.path.isdir(destination):
        fList = os.listdir(source)
        for f in fList:
            fName = os.path.join(source, f)
            if os.path.isfile(fName):
                cp(fName, destination)
        return True
    else:
        return False







# Conditional to switch between only to count files or create them
condCount = False
for arg in sys.argv:
    if arg == '-c': condCount = True

woAnnsPath = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/annotations/'
woImgPath = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/images/'
andreImgsPath = 'andrewmvd/images/'
andreAnnsPath = 'andrewmvd/annotations/'
newAnnsPath = 'guilhermecfmello/annotations/'
newImgsPath = 'guilhermecfmello/images/'


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


annsList = [f for f in listdir(woAnnsPath) if isfile(join(woAnnsPath, f))]

# List of files inserted on new database 
filesList = {}


cont = 0
if condCount:
    for ann in annsList:
        jsonFile = open(woAnnsPath+ann)
        jsonObj = json.load(jsonFile)
        cont += 1 if len(getValidAnnsXml(jsonObj, maskFilter)) > 0 else 0
    print('Total files found: ' + str(cont))
    exit()

for ann in annsList:
    jsonFile = open(woAnnsPath+ann)
    jsonObj = json.load(jsonFile)
    
    anns = getValidAnnsXml(jsonObj, maskFilter)
    if len(anns) > 1:
        cont += 1
        # getting images informations
        imgName = getImgName(jsonObj)

        im = Image.open(woImgPath+imgName)
        width, height = im.size
        # Setting new files names
        newImgName = imgName[:imgName.find('.')] + 'w' + imgName[imgName.find('.'):] if imgName.find('.') >= 0 else imgName
        newAnnName = newImgName[:newImgName.find('.')] + '.xml' if newImgName.find('.') >= 0 else newImgName + '.xml'

        # Creating new Annotation file 
        newXml = createAnn(newImgName, width, height, anns)


        # Transforming wobotdatabase to andrewmvd format
        try:
            # saving new annotation
            tree = ET.ElementTree(newXml)
            xmlStr = minidom.parseString(ET.tostring(newXml)).toprettyxml(indent='  ')
            with open(newAnnsPath+newAnnName, "w") as f:
                # printing xml as string without header 'version'
                f.write(xmlStr[23:])

            # Copying image
            im.save(newImgsPath + newImgName)
        except Exception:
            print("Error creating anns files or copying img files from wobot database")
            exit()

        # Copying andrewmvd database
        try:
            # copying imgs
            copyDirectory(andreImgsPath, newImgsPath)

            # copying annotations
            copyDirectory(andreAnnsPath, newAnnsPath)
            
            # Copying image
            im.save(newImgsPath + newImgName)
        except Exception:
            print("Error creating anns files or copying img files from andrewmvd database")
            exit()

print('Total files found: ' + str(cont))
