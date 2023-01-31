import matplotlib.pyplot as plt
import numpy as np 
import open3d as o3d

def vis(data,label):
    '''
    :param data: n*3 matrix
    :param label: n*1 matrix
    :return: visualization
    '''
    labels=np.asarray(label)
    max_label=labels.max()

    # 颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors=o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pt1],'part of cloud',width=500,height=500)