import open3d as o3d
import numpy as np

#Axis-aligned Bounding box class
class AxisAlignedBoundingBox(object):
    def __init__(self, vertices, center, size, time_step=0):
        self.time_step = time_step
        self.vertices = vertices
        self.center = center
        self.size = size

class OrientedBoundingBox(object):
    def __init__(self, vertices, center, max_bound, min_bound, time_step = 0):
        self.time_step = time_step
        self.vertices = vertices
        self.center = center
        self.center = center
        self.max_bound = max_bound
        self.min_bound = min_bound

def ExtractBBox(points,labels,method='oriented'):
    ''' Extract the axis-aligned or oriented bounding boxes from clustered point cloud
    
    Parameters
    ----------
    
    `points` (`numpy.ndarray`): nx3 clustered point cloud
      
    `method` (`string`): method used. Options: `'axisAligned'`, `'oriented'`

    Returns
    -------
    `bboxes_vertices` (`numpy.ndarray`): list of bounding boxes
    '''
    if method == 'oriented':
        bboxes_vertices = []
        bboxes = []
        sortedLabels = np.unique(labels)
        for label in sortedLabels:
            cluster = []
            for i in range(len(labels)):
                if labels[i] == label:
                    cluster.append(points[i])
            if len(cluster) < 7:   #Noise removal
                continue
            cluster = np.array(cluster)
            cluster = o3d.utility.Vector3dVector(cluster)
            bbox_vertices = o3d.geometry.OrientedBoundingBox()
            center = bbox_vertices.get_center()
            max_bound = bbox_vertices.get_max_bound()
            min_bound = bbox_vertices.get_min_bound()
            bbox_vertices = bbox_vertices.create_from_points(cluster)
            bbox_vertices = np.asarray(bbox_vertices.get_box_points())
            bbox_vertices = np.array([bbox_vertices[3],bbox_vertices[6],bbox_vertices[1],bbox_vertices[0],bbox_vertices[5],bbox_vertices[4],bbox_vertices[7],bbox_vertices[2]])
            bbox = OrientedBoundingBox(bbox_vertices, center, max_bound, min_bound)
            bboxes_vertices.append(bbox_vertices)
            bboxes.append(bbox)
        bboxes_vertices = np.array(bboxes_vertices)
        return bboxes, bboxes_vertices

    if method == 'axisAlighed':
        bboxes_vertices = []
        bboxes = []
        for label in np.unique(labels):
            cluster = points[label == labels,:]
            if len(cluster) < 7:   #Noise removal
                continue
            x_min, x_max = np.min(cluster[:, 0]), np.max(cluster[:, 0])
            y_min, y_max = np.min(cluster[:, 1]), np.max(cluster[:, 1])
            z_min, z_max = np.min(cluster[:, 2]), np.max(cluster[:, 2])
            bbox_vertices = np.array([[x_min,y_min,z_min],
                            [x_max,y_min,z_min],
                            [x_max,y_max,z_min],
                            [x_min,y_max,z_min],
                            [x_min,y_min,z_max],
                            [x_max,y_min,z_max],
                            [x_max,y_max,z_max],
                            [x_min,y_max,z_max]])
            bbox = AxisAlignedBoundingBox(bbox_vertices, \
                [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2], [x_max-x_min, y_max-y_min, z_max-z_min])
            bboxes_vertices.append(bbox_vertices)
            bboxes.append(bbox)
        bboxes_vertices = np.array(bboxes_vertices)
        return bboxes, bboxes_vertices