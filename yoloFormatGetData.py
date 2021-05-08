# SCRIPT ESCRITO PARA LEVANTAR ESTATISTICAS SOBRE OS DADOS QUE ALIMENTAM A REDE YOLO
# TAIS COMO: QUANTIDADE DE CLASSES, QUANTIDADE DE AMOSTRAS DE CADA CLASSE, ETC...
# O DIRETORIO DEVE CONTER OS ARQUIVOS DE DESCRICAO TXT NO FORMATO ACEITO PELA YOLOv4
import os
from os import listdir
from os.path import isfile, join
# Copy file
from shutil import copy as cp
# Args
import sys



sourcePath = 'guilhermecfmello/yolo-format-reduced/reduced/validation/'
resultsPath = 'datasetDescribe.txt'

classesMap = {
    0: 'without_mask',
    1: 'with_mask',
    2: 'mask_weared_incorrect'
}
#           without_mask    with_mask   mask_weared_incorrect
classesQt = [0,             0,          0]
errors = 0
executed = 0

imgsList = []
filesList = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]


def txtVerify(fileName):
    f = fileName[fileName.find('.'):]
    if f == '.txt':
        return True
    else:
        return False

for fileName in filesList:
    if txtVerify(fileName):
        f = open(sourcePath+fileName, 'r')
        line = f.readline()
        try:
            while line:
                classesQt[int(line[0])] += 1
                line = f.readline()
            executed += 1
        except:
            errors += 1
            print("Error on : " + fileName)

print(classesQt)
print("Number of errors: "  + str(errors))
print("Number of files executed: "  + str(executed))