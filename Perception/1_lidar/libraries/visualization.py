import matplotlib.pyplot as plt
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d

def vis(points,labels,bboxes,method):
    '''
    :param points: n*3 matrix
    :param labels: n*1 matrix
    :param methods: string
    :return: visualization
    '''
    if method == 'o3d':
        labels=np.asarray(labels)
        max_label=labels.max()

        # 颜色
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        pt1 = o3d.geometry.PointCloud()
        pt1.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pt1.colors=o3d.utility.Vector3dVector(colors[:, :3])

        o3d.visualization.draw_geometries([pt1],'part of cloud',width=500,height=500)
    if method == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for labels in np.unique(points[:,3]):
            ax.scatter(points[:,0:3][points[:,3] == labels, 0], 
                    points[:,0:3][points[:,3] == labels, 1], 
                    points[:,0:3][points[:,3] == labels, 2],
                    s = 3,
                    labels=str(labels))
        for bbox in bboxes:
            bbox = np.asarray(bbox.get_box_points())
            verts = [[bbox[3],bbox[6],bbox[1],bbox[0]],
            [bbox[5],bbox[4],bbox[7],bbox[2]],
            [bbox[3],bbox[6],bbox[4],bbox[5]],
            [bbox[1],bbox[0],bbox[2],bbox[7]],
            [bbox[6],bbox[1],bbox[7],bbox[4]],
            [bbox[5],bbox[2],bbox[0],bbox[3]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors=[0,0,0,0], linewidths=1, edgecolors='r'))

        ax.legend()
        plt.show()