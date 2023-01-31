import open3d as o3d
import numpy as np

def ExtractBBox(points,labels):
    bboxes = []
    sortedLabels = np.unique(labels)
    for label in sortedLabels:
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                cluster.append(points[i])
        cluster = np.array(cluster)
        cluster = o3d.utility.Vector3dVector(cluster)
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox.create_from_points(cluster)
        bboxes.append(bbox)
    return bboxes

        
