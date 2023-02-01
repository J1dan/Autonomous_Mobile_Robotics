import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
points = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])
points = np.array([[-17.96630611,  -0.64812201,  -0.0965376 ],
 [-18.20240292 , -6.50630761  , 0.41413663],
 [-17.60743309 , -0.36975271  , 3.26268373],
 [-18.41799162 , -0.62587241 , -0.05012681],
 [-18.29521543 , -6.20568871 ,  3.81976876],
 [-18.05911861 , -0.34750311 ,  3.30909453],
 [-18.65408844,  -6.48405801 ,  0.46054742],
 [-17.84352991 , -6.22793832 ,  3.77335796]])
print(np.shape(points))
bboxes = points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = [-1,1]
X, Y = np.meshgrid(r, r)
ax.scatter3D(bboxes[:, 0], bboxes[:, 1], bboxes[:, 2])
# 0-3 1-6 2-1 3-0
# 4-5 5-4 6-7 7-2
verts = [[bboxes[3],bboxes[6],bboxes[1],bboxes[0]],
 [bboxes[5],bboxes[4],bboxes[7],bboxes[2]],
 [bboxes[3],bboxes[6],bboxes[4],bboxes[5]],
 [bboxes[1],bboxes[0],bboxes[2],bboxes[7]],
 [bboxes[6],bboxes[1],bboxes[7],bboxes[4]],
 [bboxes[5],bboxes[2],bboxes[0],bboxes[3]]]


print(np.shape(verts))
ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))

plt.show()