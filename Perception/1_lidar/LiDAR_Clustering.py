import os
import sys
import pathlib
import math
import random
import json
import numpy as np 
import sklearn.cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
import open3d as o3d

# sys.path.append(str(pathlib.Path('task1_lidar_cls.ipynb').parent.parent))
from libraries.clustering import clustering
from libraries.ground_segmentation import ground_segmentation
from libraries.visualization import vis 
from libraries.BoundingBoxExtraction import ExtractBBox

file_data = np.fromfile('/home/jidan/Documents/me5413/1_lidar/lidar_data/frame2.pcd.bin', dtype=np.float32)
points = file_data.reshape((-1, 5))[:, :4]

#Ground segmentation
ground_cloud,segmented_cloud, index_ground, index_segmented = ground_segmentation(points[:,0:3],'brutal')
x = points[:, 0]  # x position of point
y = points[:, 1]  # y position of point
z = points[:, 2]  # z position of point
r = points[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
degr = np.degrees(np.arctan(z / d))
# vals = 'height'
# if vals == "height":
#     col = z
# else:
#     col = d

#Clustering
method = 'birch'  #Options: 'dbscan','kmeans','optics','meanshift','Agglomerative', 'birch'
labels = clustering(segmented_cloud, method)
labels += 1
#Add labels to the whole pointcloud
points = np.insert(points,4,0,axis = 1)
points[index_ground, 4] = 1
points[index_segmented, 4] = labels
#Add labels to the segmented pointcloud
segmented_cloud = np.insert(segmented_cloud,3,0,axis = 1)
segmented_cloud[:,3] = labels
#Noise removal
points = points[points[:,4] > 0]
segmented_cloud = segmented_cloud[segmented_cloud[:,3] > 0]

#Bounding Box Extraction
axisAligned_bboxes, axisAligned_bboxes_vertices = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'axisAlighed')
oriented_bboxes, oriented_bboxes_vertices = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'oriented')

# Visualization
vis(points[:,0:3],points[:,4],axisAligned_bboxes_vertices,method='o3d')#With ground points
vis(segmented_cloud[:,0:3],segmented_cloud[:,3],axisAligned_bboxes_vertices,method='o3d')#Without ground points
vis(segmented_cloud[:,0:3],segmented_cloud[:,3],axisAligned_bboxes_vertices,method='plt')#With axis aligned bounding boxes
vis(segmented_cloud[:,0:3],segmented_cloud[:,3],oriented_bboxes_vertices,method='plt')#With oriented bounding boxes

#Data saving
data = {}
for i in range(len(axisAligned_bboxes_vertices)):
    data[f'cluster {i}'] = axisAligned_bboxes_vertices[i].tolist()

with open('lidar_clustering.json', 'w') as f:
    json.dump(data, f)