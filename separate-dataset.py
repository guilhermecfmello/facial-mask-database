# directory navigate import
import os
from os import listdir
from os.path import isfile, join
import fnmatch
# Copy file
from shutil import copyfile as cp
# Random function
from random import shuffle


sourcePath = 'guilhermecfmello/cropped/'
sourceAnn = sourcePath + 'annotations/'
sourceImg = sourcePath + 'images/'


destinyPath = ['guilhermecfmello/cropped/test/','guilhermecfmello/cropped/validation/','guilhermecfmello/cropped/training/']
amountFiles = [993, 993, 7949]
imgFormats = ['jpg','jpeg','png']

def getImgExt(imgName, imgFormats):
    for ext in imgFormats:
        fullImageName = imgName+'.'+ext
        if isfile(fullImageName):
            return ext
    return None

annsList = [f for f in listdir(sourceAnn) if isfile(join(sourceAnn, f))]
shuffle(annsList)
print('Anns amount: ' + str(len(annsList)))
counts = [0,0,0]
for dPath, amount in zip(destinyPath, amountFiles):
    newAnns = annsList[:amount]
    annsList = annsList[amount:]
    destinyAnn = dPath + 'annotations/'
    destinyImg = dPath + 'images/'
    input("Trying: " + dPath + " Amount: " + str(len(newAnns)) + " Press some key to continue...")
    i = 0
    for ann in newAnns:
        # Copy Ann
        destinyAnnName = destinyAnn + ann.split('/')[-1]
        cp(sourceAnn+ann, destinyAnnName)
        # Copy Img
        imgName = sourceImg+ann[:len(ann)-4]
        imgName += '.'+getImgExt(imgName, imgFormats)
        destinyImageName = destinyImg+imgName.split('/')[-1]
        cp(imgName, destinyImageName)
        print("Ann and Img iterator: " + str(counts[i]))
        counts[i] += 1
    i = i + 1
      


