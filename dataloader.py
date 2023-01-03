from dataclasses import dataclass
import os
from path import Path
import numpy as np
import torch
from torchvision import transforms
import math, random
from pygem import FFD, RBF, IDW
from utils import display_pointcloud, read_off
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformations import *
import transformations




train_transform = transforms.Compose([
                                PointSampler(1024),
                                RandRotation_z(),
                                RandomNoise(),
                                Normalize(),
                                ToTensor()
                              ])

test_transform = transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])

transform = transforms.Compose([PointSampler(1024)])
transform2 = transforms.Compose([Normalize()])

class Dataset:
    def __init__(self, mode, transform):
        self.transform = transform
        self.mode = mode
        DATAPATH = os.path.abspath(os.path.join('ModelNet40'))
        folders = [dir for dir in sorted(os.listdir(Path(DATAPATH))) if os.path.isdir(Path(DATAPATH)/dir)]
        i=0
        self.files = []
        self.classes = []
        for folder in folders:
            for file in os.listdir(os.path.join(DATAPATH, folder, mode)):
                self.files.append(os.path.join(DATAPATH, folder, mode, file))
                self.classes.append(i)
            i+=1
        
    def cutmix_r(self, idx):
        idx = idx % len(self.files)
        idx2 = random.randint(0, len(self.files)-1)
        #Anzahl Punkte, die zufällig zu wählen sind
        n1 = random.randint(1, 1023)
        n2 = 1024-n1

        with open(Path(self.files[idx]), 'r') as f:
            verts1, faces1 = read_off(f)
        with open(Path(self.files[idx2]), 'r') as f:
            verts2, faces2 = read_off(f)

        transform_idx1 = transforms.Compose([PointSampler(n1)])
        transform_idx2 = transforms.Compose([PointSampler(n2)])
        transform3 = transforms.Compose([Normalize(), ToTensor()])
        pointcloud1 = transform_idx1((verts1, faces1))
        pointcloud2 = transform_idx2((verts2, faces2))

        #Die Punkte der beiden Pointclouds zusammennehmen zu einer neuen
        pointcloud = np.concatenate((pointcloud1, pointcloud2), axis=0)
        pointcloud = transform3(pointcloud)

        #Klassenzugehörigkeit festlegen: die Klasse, von der mehr Punkte enthalten sind
        percent = 1/1024*n1
        if percent < 0.5:
            cl = self.classes[idx2]
        else:
            cl = self.classes[idx]
        return pointcloud, cl
    
    def cutmix_k(self, idx):
        idx = idx % len(self.files)
        idx2 = random.randint(0, len(self.files)-1)
        rand = random.randint(1,1023)
        with open(Path(self.files[idx]), 'r') as f:
            verts1, faces1 = read_off(f)
        with open(Path(self.files[idx2]), 'r') as f:
            verts2, faces2 = read_off(f)
        transform_idx1 = transforms.Compose([
                            PointSampler(1024),
                            Normalize()
                            ])
        transform_idx2 = transforms.Compose([
                            PointSampler(1024-rand),
                            Normalize()
                            ])
        transform3 = transforms.Compose([
                            Normalize(),
                            ToTensor()
                            ])
        pointcloud1 = transform_idx1((verts1, faces1))
        pointcloud2 = transform_idx1((verts2, faces2))
        #k-nearest-Neighbours on pointcloud1:
        narray = np.array(pointcloud1)
        y = np.random.randint(0, 1024) #zufälliger Punkt der ersten Pointcloud
        y = narray[y]
        e_dist =  np.linalg.norm(narray - y, axis=1) #euklidische Distanz von y zu allen Punkten aus zweiter Pointcloud
        nearest_neighbours = np.ndarray.tolist(e_dist.argsort()[:rand]) #k-nearest-Neighbours mit k=rand-1 (y ist erste darin)
        pointcloud1 = np.zeros((len(nearest_neighbours), 3))
        j  = 0
        for i in nearest_neighbours:
            pointcloud1[j] = narray[i]
            j += 1
        #finde die 1024-rand nähesten Punkten zwischen den Punkten von den k-nearest-Neighbours und pointcloud2
        pointcloud2 = np.array(pointcloud2)
        #finde den nächsten Punkt zu Poincloud2 und bestimme dessen nächste Nachbarn
        e_dist =  np.array([np.linalg.norm(pointcloud2 - p, axis=1) for p in pointcloud1])
        e_dist = e_dist.flatten()
        nearest = np.ndarray.tolist(e_dist.argsort()[:1])
        pointcloud3 = np.zeros((len(nearest), 3))
        point = pointcloud2[(nearest[0]%1024)]
        'point oder y??'
        e_dist =  np.linalg.norm(pointcloud2 - y, axis=1) #euklidische Distanz von y zu allen Punkten aus zweiter Pointcloud
        nearest_neighbours = np.ndarray.tolist(e_dist.argsort()[:(1024-rand)]) #k-nearest-Neighbours mit k=rand-1 (y ist erste darin)
        pointcloud3 = np.zeros((len(nearest_neighbours), 3))
        j  = 0
        for i in nearest_neighbours:
            pointcloud3[j] = pointcloud2[i]
            j += 1
        #füge beide zusammen
        pointcloud = np.concatenate((pointcloud1, pointcloud3), axis=0)
        pointcloud = transform3(pointcloud)
        #bestimme Klasse
        percent = 1/1024*rand
        if percent < 0.5:
            cl = self.classes[idx2]
        else:
            cl = self.classes[idx]
        return pointcloud, cl

    def __len__(self):
            return len(self.files)
        

    def __getitem__(self, idx):
        if not idx >= len(self.files):
            with open(Path(self.files[idx]), 'r') as f:
                verts, faces = read_off(f)
            pointcloud = self.transform((verts, faces))
            
            return pointcloud, self.classes[idx]



def __main__():
    #train_ds = Dataset("train", transform=train_transform)
    #test_ds = Dataset("test", transform=test_transform)
    #train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
    #test_loader = DataLoader(dataset=test_ds, batch_size=64)
    #item = train_ds.__getitem__(train_ds.__len__()*1-200)
    #item = train_ds.__getitem__(1)
    #print(item[1])
    with open(Path(os.path.abspath(os.path.join('ModelNet40', 'vase', 'train', 'vase_0001.off' ))), 'r') as f:
        verts, faces = read_off(f)
    
    ttransform = transforms.Compose([PointSampler(1024), Normalize(), ToTensor()])
    pointcloud = ttransform((verts, faces))
    #display_pointcloud(pointcloud, point_size = 4)
    ttransform = transforms.Compose([PointSampler(524), Normalize(), (5), Normalize(), ToTensor()])
    pointcloud = ttransform((verts, faces))
    print(pointcloud.shape)
    display_pointcloud(pointcloud, point_size = 4)

    return 


if __name__ == "__main__":
    __main__()
