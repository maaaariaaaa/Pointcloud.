#Quelle: https://github.com/jiachens/ModelNet40-C

from dataclasses import dataclass
import os
from path import Path
import numpy as np
import torch
from torchvision import transforms
import math, random
from pygem import FFD, RBF, IDW
import utils
from utils import display_pointcloud
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def core_distortion(points, n_control_points=[2,2,2], displacement=None):
    """
        Ref: http://mathlab.github.io/PyGeM/tutorial-1-ffd.html
    """
    # the size of displacement matrix: 3 * control_points.shape
    if displacement is None:
        displacement = np.zeros((3,*n_control_points))

    ffd = FFD(n_control_points=n_control_points)
    ffd.box_length = [2.,2.,2.]
    ffd.box_origin = [-1., -1., -1.]
    ffd.array_mu_x = displacement[0,:,:,:]
    ffd.array_mu_y = displacement[1,:,:,:]
    ffd.array_mu_z = displacement[2,:,:,:]
    new_points = ffd(points)

    return new_points


def distortion(points, direction_mask=np.array([1,1,1]), point_mask=np.ones((5,5,5)), severity=0.5):
    n_control_points=[5,5,5]
    # random
    displacement = np.random.rand(3,*n_control_points) * 2 * severity - np.ones((3,*n_control_points)) * severity
    displacement *= np.transpose(np.tile(direction_mask, (5, 5, 5, 1)), (3, 0, 1, 2))
    displacement *= np.tile(point_mask, (3, 1, 1, 1))
    
    points = core_distortion(points, n_control_points=n_control_points, displacement=displacement)
    
    # points = denomalize(points, scale, offset)
    # set_points(data, points)
    return points

def distortion_2(points, severity=(0.4,3), func = 'gaussian_spline'):

    rbf = RBF(func=func)
    xv = np.linspace(-1, 1, severity[1]) #evenly spaced numbers in intervall -1,1, num=severity[1] samples generated
    yv = np.linspace(-1, 1, severity[1])
    zv = np.linspace(-1, 1, severity[1])
    z, y, x = np.meshgrid(zv, yv, xv) 
    mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    rbf.original_control_points = mesh
    alpha = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    gamma = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    distance = np.ones(mesh.shape[0]) * severity[0]
    displacement_x = distance * np.cos(alpha) * np.sin(gamma)
    displacement_y = distance * np.sin(alpha) * np.sin(gamma)
    displacement_z = distance * np.cos(gamma)
    displacement = np.array([displacement_x,displacement_y,displacement_z]).T
    rbf.deformed_control_points = mesh + displacement
    new_points = rbf(points)
    return new_points

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        norm_pointcloud = np.nan_to_num(norm_pointcloud)

        return  norm_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class Cutout(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        c = [(2,30), (3,30), (5,30), (7,30), (10,30)][self.severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            e_dist =  np.linalg.norm(pointcloud - picked, axis=1, keepdims=True)
            nearest = np.argpartition(e_dist, c[1], axis=0)[:c[1]]
            pointcloud = np.delete(pointcloud, nearest, axis=0)
        return pointcloud

class Shear(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [0.05, 0.1, 0.15, 0.2, 0.25][self.severity-1]
        a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
        g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

        matrix = np.array([[1,0,b],[d,1,e],[f,0,1]])
        pointcloud = np.matmul(pointcloud,matrix).astype('float32')
        return pointcloud

class Upsampling(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [N//5, N//4, N//3, N//2, N][self.severity-1]
        index = np.random.choice(N, c, replace=False)
        add = pointcloud[index] + np.random.uniform(-0.05,0.05,(c, C))
        pointcloud = np.concatenate((pointcloud,add),axis=0).astype('float32')
        return pointcloud

class Rotation(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        c = [2.5, 5, 7.5, 10, 15][self.severity-1]
        theta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
        gamma = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
        beta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.

        matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])

        new_pc = np.matmul(np.matmul(np.matmul(pointcloud,matrix_1),matrix_2),matrix_3).astype('float32')

        return new_pc

class Background_noise(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [N//45, N//40, N//35, N//30, N//20][self.severity-1]
        jitter = np.random.uniform(-1,1,(c, C))
        pointcloud = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
        return pointcloud

class Gaussian_noise(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [0.01, 0.015, 0.02, 0.025, 0.03][self.severity-1]
        jitter = np.random.normal(size=(N, C)) * c
        pointcloud = (pointcloud + jitter).astype('float32')
        #clip weggelassen, dafür muss normalisiert werden
        return pointcloud

class Rbf_distortion_inv(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][self.severity-1]
        new_pc = distortion_2(pointcloud,severity=c,func='inv_multi_quadratic_biharmonic_spline')
        return new_pc.astype('float32')

class Rbf_distortion(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][self.severity-1]
        new_pc = distortion_2(pointcloud,severity=c,func='multi_quadratic_biharmonic_spline')
        return new_pc.astype('float32')


class Ffd_distortion(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [0.1,0.2,0.3,0.4,0.5][self.severity-1]
        new_pc = distortion(pointcloud,severity=c)
        return new_pc

class Density_dec(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        c = [(1,100), (2,100), (3,100), (4,100), (5,100)][self.severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            e_dist =  np.linalg.norm(pointcloud - picked, axis=1, keepdims=True)
            nearest = np.argpartition(e_dist, c[1], axis=0)[:c[1]]
            idx = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
            nearest = nearest[idx]
            pointcloud = np.delete(pointcloud, nearest, axis=0)
        return pointcloud#
    
    def __call__(self, pointcloud):
        c = [(2,30), (3,30), (5,30), (7,30), (10,30)][self.severity-1]
        for _ in range(c[0]):
            i = np.random.choice(pointcloud.shape[0],1)
            picked = pointcloud[i]
            e_dist =  np.linalg.norm(pointcloud - picked, axis=1, keepdims=True)
            nearest = np.argpartition(e_dist, c[1], axis=0)[:c[1]]
            pointcloud = np.delete(pointcloud, nearest, axis=0)
        return pointcloud

class Impulse_noise(object):
    def __init__(self, severity):
        assert isinstance(severity, int)
        self.severity = severity

    def __call__(self, pointcloud):
        N, C = pointcloud.shape
        c = [N//30, N//25, N//20, N//15, N//10][self.severity-1]
        index = np.random.choice(N, c, replace=False)
        pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.1
        return pointcloud

# class Density_inc(object):
#     def __init__(self, severity):
#         assert isinstance(severity, int)
#         self.severity = severity

#     def __call__(self, pointcloud):
#         N, C = pointcloud.shape
#         c = [(1,100), (2,100), (3,100), (4,100), (5,100)][self.severity-1]
#         # idx = np.random.choice(N,c[0])
#         temp = []
#         for _ in range(c[0]):
#             i = np.random.choice(pointcloud.shape[0],1)
#             picked = pointcloud[i]
#             e_dist =  np.linalg.norm(pointcloud - picked, axis=1, keepdims=True)
#             nearest = np.argpartition(e_dist, c[1], axis=0)[:c[1]]
#             #idx = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
#             #nearest = nearest[idx]
#             add = pointcloud[nearest.squeeze()] + np.random.uniform(-0.05,0.05,(c[1], C))
#             temp.append(add)
#             temp.append(pointcloud[nearest.squeeze()])
#             pointcloud = np.delete(pointcloud, nearest.squeeze(), axis=0)
        
#         #idx = np.random.choice(pointcloud.shape[0],N - c[0] * c[1])
#         temp.append(pointcloud)
#         pointcloud = np.concatenate(temp)
#         return pointcloud

# class Uniform_downsampling(object):
#     def __init__(self, severity):
#         assert isinstance(severity, int)
#         self.severity = severity

#     def __call__(self, pointcloud):
#         N, C = pointcloud.shape
#         c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][self.severity-1]
#         index = np.random.choice(N, N - c, replace=False)
#         return pointcloud[index]


