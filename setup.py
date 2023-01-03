import os
from path import Path
import numpy as np
import shutil
from utils import create_dirs

def move(files, classes, classnum, movenum, path):
    DATAPATH = os.path.abspath(os.path.join('ModelNet40'))
    idx = np.where(classes == classnum)[0]
    files_0 = files[idx]
    random_idx = np.random.choice(len(files_0), movenum, replace=False)
    files_0 = files_0[random_idx]
    create_dirs(os.path.join(DATAPATH, path, 'val'))
    for i in files_0:
        shutil.move(i, os.path.join(DATAPATH, path, 'val'))

def setup():
    DATAPATH = os.path.abspath(os.path.join('ModelNet40'))
    folders = [dir for dir in sorted(os.listdir(Path(DATAPATH))) if os.path.isdir(Path(DATAPATH)/dir)]
    i=0
    files = []
    classes = []
    for folder in folders:
        for file in os.listdir(os.path.join(DATAPATH, folder, 'train')):
            files.append(os.path.join(DATAPATH, folder, 'train', file))
            classes.append(i)
        i+=1
    files = np.array(files)
    classes = np.array(classes)

    i = 0
    files_test = []
    classes_test = []
    for folder in folders:
        for file in os.listdir(os.path.join(DATAPATH, folder, 'test')):
            files_test.append(os.path.join(DATAPATH, folder, 'test', file))
            classes_test.append(i)
        i+=1
    files_test = np.array(files_test)
    classes_test = np.array(classes_test)

    move(files, classes, 0, 100, 'airplane')
    move(files_test, classes_test, 1, 25, 'bathtub')
    move(files, classes, 2, 100, 'bed' )
    move(files, classes, 3, 20, 'bench')
    move(files, classes, 4, 100, 'bookshelf')
    move(files_test, classes_test, 5, 50, 'bottle')
    move(files_test, classes_test, 6, 7, 'bowl')
    move(files, classes, 6, 6, 'bowl')
    move(files_test, classes_test, 7, 50, 'car')
    move(files, classes, 8, 100, 'chair')
    move(files, classes, 9, 20, 'cone')
    move(files_test, classes_test, 10, 7, 'cup')
    move(files, classes, 10, 6, 'cup')
    move(files, classes, 11, 20, 'curtain')
    move(files_test, classes_test, 12, 43, 'desk')
    move(files, classes, 13, 20, 'door')
    move(files_test, classes_test, 14, 43, 'dresser')
    move(files, classes, 15, 20, 'flower_pot')
    move(files_test, classes_test, 16, 50, 'glass_box')
    move(files_test, classes_test, 17, 50, 'guitar')
    move(files, classes, 18, 20, 'keyboard')
    move(files, classes, 19, 20, 'lamp')
    move(files, classes, 20, 20, 'laptop')
    move(files_test, classes_test, 21, 50, 'mantel')
    move(files_test, classes_test, 22, 15, 'monitor')
    move(files, classes, 22, 70, 'monitor')
    move(files_test, classes_test, 23, 43, 'night_stand')
    move(files_test, classes_test, 24, 5, 'person')
    move(files, classes, 24, 10, 'person')
    move(files_test, classes_test, 25, 50, 'piano')
    move(files_test, classes_test, 26, 50, 'plant')
    move(files, classes, 27, 20, 'radio')
    move(files_test, classes_test, 28, 50, 'range_hood')
    move(files, classes, 29, 20, 'sink')
    move(files, classes, 30, 100, 'sofa')
    move(files, classes, 31, 20, 'stairs')
    move(files, classes, 32, 14, 'stool')
    move(files_test, classes_test, 32, 3, 'stool')
    move(files_test, classes_test, 33, 15, 'table')
    move(files, classes, 33, 60, 'table')
    move(files, classes, 34, 20, 'tent')
    move(files_test, classes_test, 35, 30, 'toilet')
    move(files, classes, 35, 40, 'toilet')
    move(files_test, classes_test, 36, 50, 'tv_stand')
    move(files, classes, 37, 100, 'vase')
    move(files, classes, 38, 10, 'wardrobe')
    move(files_test, classes_test, 38, 5, 'wardrobe')
    move(files, classes, 39, 20, 'xbox')   

def __main__():
    #setup()
    return 


if __name__ == "__main__":
    __main__()