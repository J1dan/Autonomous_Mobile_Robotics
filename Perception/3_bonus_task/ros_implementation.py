import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from libraries.clustering import clustering
from libraries.ground_segmentation import ground_segmentation 
from libraries.BoundingBoxExtraction import ExtractBBox

class PointCloudProcesser(object):
    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("me5413/lidar_top",
                                     PointCloud2,
                                     self.callback, queue_size=10)

        self.marker_publisher = rospy.Publisher('bounding_box_marker', MarkerArray, queue_size=10)
        self.i = 0

    def callback(self, msg):
        self.i+=1
        if self.i != 3:
            return
        else:
            self.i = 0
        
        points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))
        points = np.array(points)

        #Ground segmentation
        _ ,segmented_cloud, _, _ = ground_segmentation(points[:,0:3],'brutal')

        #Clustering
        labels = clustering(segmented_cloud, 'dbscan')
        #Add labels to the segmented pointcloud
        segmented_cloud = np.insert(segmented_cloud,3,0,axis = 1)
        segmented_cloud[:,3] = labels
        #Noise removal
        # points = points[points[:,3] > 0]
        segmented_cloud = segmented_cloud[segmented_cloud[:,3] > -1]

        #Bounding Box Extraction
        # _, bboxes_vertices = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'axisAlighed')
        _, bboxes_vertices = ExtractBBox(segmented_cloud[:,0:3], segmented_cloud[:,3],'oriented')

        # Define the marker message
        marker_array = MarkerArray()
        for i in range(len(bboxes_vertices)):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = 'lidar_top'
            marker.type = marker.LINE_LIST
            marker.action = marker.ADD
            marker.scale.x = 0.1 # Line width
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            orientation = Quaternion()
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            marker.pose.orientation = orientation
            #lines include each edge connected by the two vertices, thus 12 edges x 2 vertices = 24 vertices needed
            line = np.array([bboxes_vertices[i][0] , bboxes_vertices[i][1] , \
                             bboxes_vertices[i][1] , bboxes_vertices[i][2] , \
                             bboxes_vertices[i][2] , bboxes_vertices[i][3] , \
                             bboxes_vertices[i][3] , bboxes_vertices[i][0] , \
                             bboxes_vertices[i][0] , bboxes_vertices[i][4] , \
                             bboxes_vertices[i][4] , bboxes_vertices[i][5] , \
                             bboxes_vertices[i][5] , bboxes_vertices[i][6] , \
                             bboxes_vertices[i][6] , bboxes_vertices[i][7] , \
                             bboxes_vertices[i][7] , bboxes_vertices[i][4] , \
                             bboxes_vertices[i][3] , bboxes_vertices[i][7] , \
                             bboxes_vertices[i][2] , bboxes_vertices[i][6] , \
                             bboxes_vertices[i][1] , bboxes_vertices[i][5]])
            lines = [Point(x, y, z) for x, y, z in line]
            marker.points = lines
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

if __name__ =='__main__':
    rospy.init_node("pointcloud_processor")
    PointCloudProcesser()
    rospy.spin()

