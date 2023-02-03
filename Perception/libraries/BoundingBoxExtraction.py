import open3d as o3d
import numpy as np

def ExtractBBox(points,labels,method='oriented'):
    ''' Extract the axis-aligned or oriented bounding boxes from clustered point cloud
    
    Parameters
    ----------
    
    `points` (`numpy.ndarray`): nx3 clustered point cloud
      
    `method` (`string`): method used. Options: `'axisAligned'`, `'oriented'`

    Returns
    -------
    `bboxes` (`numpy.ndarray`): list of bounding boxes
    '''
    if method == 'oriented':
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
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox = bbox.create_from_points(cluster)
            bbox = np.asarray(bbox.get_box_points())
            bbox = np.array([bbox[3],bbox[6],bbox[1],bbox[0],bbox[5],bbox[4],bbox[7],bbox[2]])
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        return bboxes

    if method == 'axisAlighed':
        bboxes = []
        for label in np.unique(labels):
            cluster = points[label == labels,:]
            if len(cluster) < 7:   #Noise removal
                continue
            x_min, x_max = np.min(cluster[:, 0]), np.max(cluster[:, 0])
            y_min, y_max = np.min(cluster[:, 1]), np.max(cluster[:, 1])
            z_min, z_max = np.min(cluster[:, 2]), np.max(cluster[:, 2])
            bbox = np.array([[x_min,y_min,z_min],
                            [x_max,y_min,z_min],
                            [x_max,y_max,z_min],
                            [x_min,y_max,z_min],
                            [x_min,y_min,z_max],
                            [x_max,y_min,z_max],
                            [x_max,y_max,z_max],
                            [x_min,y_max,z_max]])
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        return bboxes