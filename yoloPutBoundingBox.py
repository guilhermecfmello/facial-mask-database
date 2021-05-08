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
from matplotlib.image import imread
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

classesMap = {
    0 : 'without_mask',
    1 : 'with_mask',
    2 : 'mask_weared_incorrect',
}

colorsMap = {
    0 : 'magenta',
    1 : 'darkorange',
    2 : 'g'
}

sourcePath = 'guilhermecfmello/yolo-format-90-10/validation/'
destinyPath = 'guilhermecfmello/yolo-format-bnd-test/'

fileList = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]

# return xMin, yMin, width, height
def getBndBox(imgWidth, imgHeight, xCenter, yCenter, width, height):
    xC = xCenter * imgWidth
    yC = yCenter * imgHeight
    widthBndBox = imgWidth * width
    heightBndBox = imgHeight * height
    xMin = xC - (widthBndBox/2)
    yMin = yC - (heightBndBox/2)
    return xMin, yMin, widthBndBox, heightBndBox

# write rectangle on image
def writeBndBox(img, bndbox, bndClass, ax):
    xMin = bndbox[0]
    yMin = bndbox[1]
    width = bndbox[2]
    height = bndbox[3]
    color = colorsMap[bndClass]
    rect = patches.Rectangle((xMin, yMin), width, height,linewidth=1,edgecolor=color,facecolor='none')
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.add_patch(rect)
    # ax.annotate(rect, xMin, yMin)
    return ax


for f in fileList:
    generalName = f[:f.find('.')]
    #  Fazer para cada arquivo txt
    if f[f.find('.'):] != '.txt':
        img = np.array(Image.open(sourcePath+f), dtype=np.uint8)
 
        h = np.shape(img)[0]
        w = np.shape(img)[1]

        txt = open(sourcePath+generalName+'.txt', 'r')
        bndbox = txt.readline()
        ax = None
        while bndbox:
            # print("bndbox: " + str(bndbox))
            bnd = bndbox.split(' ')
            bndClass = int(bnd[0])
            xCenter = float(bnd[1])
            yCenter = float(bnd[2])
            bndWidth = float(bnd[3])
            bndHeight = float(bnd[4])
            newBndBox = getBndBox(w, h, xCenter, yCenter, bndWidth, bndHeight)
            ax = writeBndBox(img, newBndBox, bndClass, ax)
            # preparar bounding box aqui
            bndbox = txt.readline()
        plt.savefig(destinyPath+f)


