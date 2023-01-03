import os
from path import Path
from torchvision import transforms
from dataloader import PointSampler, Normalize, read_off
import matplotlib.pyplot as plt

#Transformationen
transform = transforms.Compose([PointSampler(1024)])
transform2 = transforms.Compose([Normalize()])

#Plotte das Histogramm von den x-,y- undz-Koordinaten der Pointcloud-Punkte
def make_hist(pointcloud):
    x = pointcloud[:,0]
    y = pointcloud[:,1]
    z = pointcloud[:,2]

    plt.hist(x, bins=50, alpha=0.5, label='x')
    plt.hist(y, bins=50, alpha=0.5, label='y')
    plt.hist(z, bins=50, alpha=0.5, label='z')
    plt.legend(loc='upper right')
    plt.show()

def __main__():
    #was visualisiert werden soll
    with open(Path(os.path.abspath(os.path.join('ModelNet40', 'vase', 'train', 'vase_0001.off' ))), 'r') as f:
        verts, faces = read_off(f)

    #visualisiere nicht normalisierte x-, y-, z-Koordinaten in Histogramm
    pointcloud = transform((verts,faces)) 
    make_hist(pointcloud)

    #visualisiere normaliesierte x-, y-, z-Koordinaten im Histogramm
    pointcloud = transform2(pointcloud)
    make_hist(pointcloud)
    return 


if __name__ == "__main__":
    __main__()
