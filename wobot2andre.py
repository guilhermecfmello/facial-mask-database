# xml import
import xml.etree.ElementTree as ET
# json import
import json
# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch
# Image
from PIL import Image

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
def getValidAnns(obj, maskFilter):
    print(str(len(obj['Annotations'])))
    anns = ET.Element('object')
    
    for ann in obj['Annotations']:
        if annIsValid(ann, maskFilter):
            newAnn = {
                'name' : maskFilter[ann['classname']],
                'pose' : 'Unspecified',
                'truncated' : str(0),
                'occluded' : str(0),
                'difficult' : str(0),
                'bndbox': {
                    'xmin' : str(999),
                    'ymin' : str(999),
                    'xmax' : str(999),
                    'ymax' : str(999)
                }
            }
            print(newAnn)
            print("============================")

def createAnn(fileName, width, height):
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text = 'images'
    ET.SubElement(ann, 'filename').text = fileName

    size = ET.SubElement(ann, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(3)

    ET.SubElement(ann, 'segmented').text = str(0)
    return ann

def insertAnnObj(ann):
    pass


def dictMerge(dict1, dict2):
      dict3 = dict1.copy()   # start with x's keys and values
      dict3.update(dict2)    # modifies z with y's keys and values & returns None
      return dict3
# className to filter images




path = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/'
anns = 'annotations/'
imgs = 'images/'

# fullPath = 
annsList = [f for f in listdir(path + anns) if isfile(join(path + anns, f))]

# List of files inserted on new database 
filesList = {}

cont = 0

for ann in annsList:
    jsonFile = open(path+anns+ann)
    jsonObj = json.load(jsonFile)
    if hasValidClass(jsonObj, maskFilter):
        # cont = cont + 1
        imgName = getImgName(jsonObj)
        # get dimensions image
        im = Image.open(path+imgs+imgName)
        width, height = im.size
        # print("width:" + str(width) + "height: " + str(height))
        newXml = createAnn("test.xml", width, height)
        anns = getValidAnns(jsonObj, maskFilter)
        exit()
        # tree = ET.ElementTree(newAnn)
        # tree.write("filename.xml")
        # exit()
        # print(str(width)+ '   ' + str(height) + '\n')
    # data = json.load(jsonFile)
    # for mask in data['Annotations']:
    #     if mask['classname'] in maskFilter:
    #         f = data['FileName']
    #         i = f.find('.')
    #         imgName = 'maksssksksssW' + f
    #         annName = 'maksssksksssW' + f[:i] + '.xml'
    #         
    #         exit()

    #         newAnn = createAnn(imgName, width, height, mask, maskFilter)
    #         print(newAnn)
    #         exit()
        # if mask['classname'] == 'hood':

            # print(jsonFile)
            # cont = cont + 1
            # if cont > 10:
                # exit()
    # print(data)
    # exit()

print('valid class amount:' + str(cont))
# print(len(annsList))
