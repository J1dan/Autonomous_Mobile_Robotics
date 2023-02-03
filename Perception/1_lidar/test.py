import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import json
import os
# points = np.array([[-1, -1, -1],
#                   [1, -1, -1 ],
#                   [1, 1, -1],
#                   [-1, 1, -1],
#                   [-1, -1, 1],
#                   [1, -1, 1 ],
#                   [1, 1, 1],
#                   [-1, 1, 1]])
# points = np.array([[-17.96630611,  -0.64812201,  -0.0965376 ],
#  [-18.20240292 , -6.50630761  , 0.41413663],
#  [-17.60743309 , -0.36975271  , 3.26268373],
#  [-18.41799162 , -0.62587241 , -0.05012681],
#  [-18.29521543 , -6.20568871 ,  3.81976876],
#  [-18.05911861 , -0.34750311 ,  3.30909453],
#  [-18.65408844,  -6.48405801 ,  0.46054742],
#  [-17.84352991 , -6.22793832 ,  3.77335796]])
# print(np.shape(points))
# bbox = points
# # 0-3 1-6 2-1 3-0
# # 4-5 5-4 6-7 7-2
# bbox = np.array([bbox[3],bbox[6],bbox[1],bbox[0],bbox[5],bbox[4],bbox[7],bbox[2]])
# print(bbox)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# r = [-1,1]
# X, Y = np.meshgrid(r, r)
# ax.scatter3D(bbox[:, 0], bbox[:, 1], bbox[:, 2])

# # verts = [[bbox[3],bbox[6],bbox[1],bbox[0]],
# #  [bbox[5],bbox[4],bbox[7],bbox[2]],
# #  [bbox[3],bbox[6],bbox[4],bbox[5]],
# #  [bbox[1],bbox[0],bbox[2],bbox[7]],
# #  [bbox[6],bbox[1],bbox[7],bbox[4]],
# #  [bbox[5],bbox[2],bbox[0],bbox[3]]]

# verts = [[bbox[0],bbox[1],bbox[2],bbox[3]],
# [bbox[4],bbox[5],bbox[6],bbox[7]],
# [bbox[0],bbox[1],bbox[5],bbox[4]],
# [bbox[2],bbox[3],bbox[7],bbox[6]],
# [bbox[1],bbox[2],bbox[6],bbox[5]],
# [bbox[4],bbox[7],bbox[3],bbox[0]]]


# print(np.shape(verts))
# ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))

# plt.show()

with open('Perception/1_lidar/lidar_clustering.json', 'r') as jsonfile:
    json_string = json.load(jsonfile)

print(json_string)