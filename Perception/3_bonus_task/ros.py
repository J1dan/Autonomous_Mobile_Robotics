import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np

class PointCloudSubscriber(object):
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("me5413/lidar_top",
                                     PointCloud2,
                                     self.callback, queue_size=10)
    def callback(self, msg):
        assert isinstance(msg, PointCloud2)
        
        points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))

        print(points[0][0])


if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()