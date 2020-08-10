# xml import
import xml.etree.ElementTree as ET
# import requests
# json import
import json
# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch


# className to filter images

maskFilter = {
    'other':'with_mask', # Needs pos treatment
    'scarf_bandana':'with_mask', # Needs pos treatment
    'face_other_covering':'with_mask', # Needs pos treatment
    'face_with_mask':'with_mask',
    'face_with_mask_incorrect':'mask_weared_incorrect',
    'face_no_mask':'without_mask',
    'mask_surgical':'with_mask',
    'mask_colorful' : 'with_mask' 
}


path = 'wobotintelligence/Medical-mask/Medical-mask/Medical-Mask/'
anns = 'annotations/'
imgs = 'images/'

fullPath = path + anns
annsList = [f for f in listdir(fullPath) if isfile(join(fullPath, f))]

cont = 0
for ann in annsList:
    jsonFile = open(fullPath+ann)
    data = json.load(jsonFile)
    classes = []
    for mask in data['Annotations']:
        if mask['classname'] not in classes:

        # if mask['classname'] == "hood":

            # print(jsonFile)
            # cont = cont + 1
            # if cont > 10:
                # exit()
    # print(data)
    # exit()

print("Others class amount " + str(cont))
# print(len(annsList))