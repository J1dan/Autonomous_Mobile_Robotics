import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import os
import sys
import pathlib
import open3d as o3d
import matplotlib.pyplot as plt
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from libraries.clustering import clustering
from libraries.ground_segmentation import ground_segmentation
from libraries.visualization import vis 

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
        ground_cloud, segmented_cloud, index_ground, index_segmented = ground_segmentation(points)

        labels = clustering(segmented_cloud,'dbscan')
        points = np.insert(points,3,0,axis = 1)
        points[index_ground, 3] = 80
        points[index_segmented, 3] = labels
        vis(points[:,0:3],points[:,3])

if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()

