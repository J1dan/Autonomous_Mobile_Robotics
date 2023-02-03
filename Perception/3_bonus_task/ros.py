import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import os
import sys
import pathlib
import open3d as o3d
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from libraries.clustering import clustering
from libraries.ground_segmentation import ground_segmentation
from libraries.visualization import vis 
from libraries.BoundingBoxExtraction import ExtractBBox

class PointCloudSubscriber(object):
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("me5413/lidar_top",
                                     PointCloud2,
                                     self.callback, queue_size=10)
    def callback(self, msg):
        assert isinstance(msg, PointCloud2)
        
        points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))
        points = np.array(points)

        #Ground segmentation
        ground_cloud,segmented_cloud, index_ground, index_segmented = ground_segmentation(points[:,0:3],'brutal')
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2]  # z position of point
        d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
        degr = np.degrees(np.arctan(z / d))
        # vals = 'height'
        # if vals == "height":
        #     col = z
        # else:
        #     col = d

        #Clustering
        method = 'dbscan'  #Options: 'dbscan','kmeans','optics','meanshift','AgglomerativeClustering', 'birch'
        labels = clustering(segmented_cloud, method)
        labels += 1
        #Add labels to the whole pointcloud
        points = np.insert(points,3,0,axis = 1)
        points[index_ground, 3] = 1
        points[index_segmented, 3] = labels
        #Add labels to the segmented pointcloud
        segmented_cloud = np.insert(segmented_cloud,3,0,axis = 1)
        segmented_cloud[:,3] = labels
        #Noise removal
        points = points[points[:,3] > 0]
        segmented_cloud = segmented_cloud[segmented_cloud[:,3] > 0]

        #Bounding Box Extraction
        axis_aligned_bboxes = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'axisAlighed')
        oriented_bboxes = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'oriented')

        #Visualization
        vis(points[:,0:3],points[:,3],axis_aligned_bboxes,method='o3d')#With ground points
        vis(segmented_cloud[:,0:3],segmented_cloud[:,3],axis_aligned_bboxes,method='o3d')#Without ground points
        # vis(segmented_cloud[:,0:3],segmented_cloud[:,3],axis_aligned_bboxes,method='plt')#With axis aligned bounding boxes
        # vis(segmented_cloud[:,0:3],segmented_cloud[:,3],oriented_bboxes,method='plt')#With oriented bounding boxes


if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()

