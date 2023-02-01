import open3d as o3d
import numpy as np

def ExtractBBox(points,labels,method):
    if method == 'oriented':
        bboxes = []
        sortedLabels = np.unique(labels)
        for label in sortedLabels:
            cluster = []
            for i in range(len(labels)):
                if labels[i] == label:
                    cluster.append(points[i])
            if len(cluster) < 4:   #Noise removal
                continue
            cluster = np.array(cluster)
            cluster = o3d.utility.Vector3dVector(cluster)
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox = bbox.create_from_points(cluster)
            bbox = np.asarray(bbox.get_box_points())
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        return bboxes
        
    if method == 'axisAlighed':
        bboxes = []
        for label in np.unique(labels):
            cluster = points[label == labels,:]
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
                

        
