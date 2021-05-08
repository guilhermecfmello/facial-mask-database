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
from shutil import move as mv
from shutil import rmtree as rmTree
# Args
import sys
# Time for progress bar
import time

N = 273
sourcePath = 'guilhermecfmello/yolo-format-reduced/'
destiny = sourcePath + 'training/'

def txtVerify(fileName):
    f = fileName[fileName.find('.'):]
    if f == '.txt':
        return True
    else:
        return False

def imgVerify(fileName):
    f = fileName[fileName.find('.'):]
    if f == '.jpg':
        return True
    else:
        return False

# Return the class from line
def getClass(line):
    if line[0] in ['0', '1', '2']:
        return int(line[0])
    return -1

# True if the file has the class without_mask
def hasWithoutMask(f):
    line = f.readline()
    qtd = 0
    while line:
        c = getClass(line)
        if c == 0:
            qtd += 1
        line = f.readline()
    f.seek(0)
    return qtd


def hasWithMask(f):
    line = f.readline()
    qtd = 0
    while line:
        c = getClass(line)
        if c == 1:
            qtd += 1
        line = f.readline()
    f.seek(0)
    return qtd

def hasMaskIncorrect(fifle):
    line = f.readline()
    qtd = 0
    while line:
        c = getClass(line)
        if c == 2:
            qtd += 1
        line = f.readline()
    f.seek(0)
    return qtd
            # without_mask, with_mask, mask_weared_incorrect
qtd =  [N,          N,       N]

filesList = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]
# filesList = ['1827w.jpg']
print("Getting mask_weared_incorrect...")
# First we get all the mask_weared_incorrect from the database
for imgName in filesList:
    if imgVerify(imgName):
        txtName = imgName[:imgName.find('.')] + '.txt'
        f = open(sourcePath+txtName, 'r')
        maskIncorrect = hasMaskIncorrect(f)
        if maskIncorrect > 0:
            withMask = hasWithMask(f)
            withoutMask = hasWithoutMask(f)
            qtd[0] -= withoutMask
            qtd[1] -= withMask
            qtd[2] -= maskIncorrect
            cp(sourcePath+imgName, destiny+imgName)
            cp(sourcePath+txtName, destiny+txtName)


# without_mask
if qtd[0] > 0:
    for imgName in filesList:
        if imgVerify(imgName):
            txtName = imgName[:imgName.find('.')] + '.txt'
            f = open(sourcePath+txtName, 'r')
            withoutMask = hasWithoutMask(f)
            withMask = hasWithMask(f)
            if withoutMask > 0 and withMask == 0:
                qtd[0] -= withoutMask
                cp(sourcePath+imgName, destiny+imgName)
                cp(sourcePath+txtName, destiny+txtName)
            if qtd[0] <= 0:
                break

print(qtd)
print("reducing finished, qtd of each class below: ")
print("without_mask: " + str(N + ((N-qtd[0])*-1)))
print("with_mask: " + str(N + ((N-qtd[1])*-1)))
print("mask_weared_incorrect: " + str(N + ((N-qtd[2])*-1)))

                



