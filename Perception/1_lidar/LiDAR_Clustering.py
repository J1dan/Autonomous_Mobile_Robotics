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
    ground_cloud,segmented_cloud, index_ground, index_segmented = ground_segmentation(points[:,0:3],'brutal')

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
labels += 1
points = np.insert(points,4,0,axis = 1)
points[index_ground, 4] = 1
points[index_segmented, 4] = labels
#Noise removal
points = points[points[:,4] > 0]

#Bounding Box Extraction
bboxes = ExtractBBox(segmented_cloud, labels)
# o3d.visualization.draw_geometries(bboxes)

#Visualization with ground points
vis(points[:,0:3],points[:,4])
#Visualization with ground points
segmented_cloud = np.insert(segmented_cloud,3,0,axis = 1)
segmented_cloud[:,3] = labels
segmented_cloud = segmented_cloud[segmented_cloud[:,3] > 0]
vis(segmented_cloud[:,0:3],segmented_cloud[:,3])

#Data saving
