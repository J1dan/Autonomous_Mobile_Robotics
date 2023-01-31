import os
import sys
import pathlib
import math
import random
import json
import numpy as np 
import open3d as o3d
import sklearn.cluster
from sklearn import metrics

# sys.path.append(str(pathlib.Path('task1_lidar_cls.ipynb').parent.parent))
from libraries.clustering import clustering
from libraries.ground_segmentation import ground_segmentation
from libraries.visualization import vis 
from libraries.BoundingBoxExtraction import ExtractBBox

file_data = np.fromfile('/home/jidan/Documents/me5413/1_lidar/lidar_data/frame2.pcd.bin', dtype=np.float32)
points = file_data.reshape((-1, 5))[:, :4]
print(np.shape(points))
# Points: (x, y, z, intensity)

GROUND_SEGMENTATION = True
if GROUND_SEGMENTATION:
    ground_cloud,segmented_cloud, index_ground, index_segmented = ground_segmentation(points[:,0:3])

x = points[:, 0]  # x position of point
y = points[:, 1]  # y position of point
z = points[:, 2]  # z position of point
r = points[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
degr = np.degrees(np.arctan(z / d))
 
vals = 'height'
if vals == "height":
    col = z
else:
    col = d

#Clustering
method = 'dbscan'  #Options: 'dbscan','kmeans','optics','meanshift','AgglomerativeClustering', 'birch'
labels = clustering(segmented_cloud, method)

#Bounding Box Extraction
bboxes = ExtractBBox(segmented_cloud, labels)
o3d.visualization.draw_geometries(bboxes)

#Visualization
vis(segmented_cloud,labels)

#Data saving
points = np.insert(points,4,0,axis = 1)
print(f"shape of points = {np.shape(points)}")

points[index_ground, 4] = 80

points[index_segmented, 4] = labels

vis(points[:,0:3],points[:,4])

points = points.tolist()

with open('lidar_clustering.json', 'w') as f:
    json.dump(points, f)