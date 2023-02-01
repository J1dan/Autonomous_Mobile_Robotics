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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d

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
bboxes = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'oriented')
# o3d.visualization.draw_geometries(bboxes)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in np.unique(segmented_cloud[:,3]):
    ax.scatter(segmented_cloud[:,0:3][segmented_cloud[:,3] == label, 0], 
               segmented_cloud[:,0:3][segmented_cloud[:,3] == label, 1], 
               segmented_cloud[:,0:3][segmented_cloud[:,3] == label, 2],
               s = 3,
               label=str(label))

# for label in np.unique(segmented_cloud[:,3]):
#     cluster = segmented_cloud[:,0:3][segmented_cloud[:,3] == label]
#     if len(cluster) < 5:
#         continue
#     x_min, x_max = np.min(cluster[:, 0]), np.max(cluster[:, 0])
#     y_min, y_max = np.min(cluster[:, 1]), np.max(cluster[:, 1])
#     z_min, z_max = np.min(cluster[:, 2]), np.max(cluster[:, 2])
    
#     xs = [x_min, x_min, x_max, x_max, x_min]
#     ys = [y_min, y_max, y_max, y_min, y_min]
#     zs = [z_min, z_min, z_min, z_min, z_min]
#     ax.plot(xs, ys, zs, label=str(label))
    
#     xs = [x_min, x_min, x_max, x_max, x_min]
#     ys = [y_min, y_max, y_max, y_min, y_min]
#     zs = [z_max, z_max, z_max, z_max, z_max]
#     ax.plot(xs, ys, zs, label=str(label))
    
#     xs = [x_min, x_min, x_min, x_min, x_min]
#     ys = [y_min, y_max, y_max, y_min, y_min]
#     zs = [z_min, z_min, z_max, z_max, z_min]
#     ax.plot(xs, ys, zs, label=str(label))
    
#     xs = [x_max, x_max, x_max, x_max, x_max]
#     ys = [y_min, y_max, y_max, y_min, y_min]
#     zs = [z_min, z_min, z_max, z_max, z_min]
#     ax.plot(xs, ys, zs, label=str(label))

for bbox in bboxes:
    verts = [[bbox[3],bbox[6],bbox[1],bbox[0]],
    [bbox[5],bbox[4],bbox[7],bbox[2]],
    [bbox[3],bbox[6],bbox[4],bbox[5]],
    [bbox[1],bbox[0],bbox[2],bbox[7]],
    [bbox[6],bbox[1],bbox[7],bbox[4]],
    [bbox[5],bbox[2],bbox[0],bbox[3]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=[0,0,0,0], linewidths=1, edgecolors='r'))

ax.legend()
plt.show()

# #Visualization with ground points
# vis(points[:,0:3],points[:,4])
# #Visualization with ground points
# segmented_cloud = np.insert(segmented_cloud,3,0,axis = 1)
# segmented_cloud[:,3] = labels
# segmented_cloud = segmented_cloud[segmented_cloud[:,3] > 0]
# vis(segmented_cloud[:,0:3],segmented_cloud[:,3])

#Data saving
