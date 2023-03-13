import numpy as np
import open3d as o3d
import time
import argparse

parser = argparse.ArgumentParser()

# Define the 'task' argument
parser.add_argument('--task', choices=['task1', 'task2'], default='task1', help='Options: task1, task2')

# Define the 'method' argument
if parser.parse_known_args()[0].task == 'task1':
    parser.add_argument('--method', choices=['none', 'downSampling'], default='none', help="Options: none, downSampling")
elif parser.parse_known_args()[0].task == 'task2':
    parser.add_argument('--method', choices=['none', 'downSampling', 'globalReg', 'combined'], default='none', help="Options: none, downSampling, globalReg, combined")

# Parse the arguments
task = parser.parse_args().task
method = parser.parse_args().method

# Downsample
def preprocess_point_cloud(pcd, voxel_size):
    ''' Perform down-sampling to the point cloud
    
    Parameters
    ----------
    
    `pcd` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): point cloud defined by Open3d library

    `voxel_size` (`float`): the parameter

    Returns
    -------
    `pcd_down` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): the downsampled point cloud
    `pcd_fpfh` (`<class 'open3d.cuda.pybind.pipelines.registration.Feature'>`): the calculated FPFH features of the point cloud
    '''
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    ''' Execute fast global registration to the down-sampled point cloud with FPFH features
    
    Parameters
    ----------
    
    `source_down` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): down-sampled source point cloud
    `target_down` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): down-sampled target point cloud
    `target_fpfh` (`<class 'open3d.cuda.pybind.pipelines.registration.Feature'>`): calculated FPFH features of source point cloud
    `target_fpfh` (`<class 'open3d.cuda.pybind.pipelines.registration.Feature'>`): calculated FPFH features of target point cloud
    `voxel_size` (`float`): the parameter

    Returns
    -------
    `regResult` (`<class 'open3d.cuda.pybind.pipelines.registration.RegistrationResult'>`): the registration result
    '''
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    regResult = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return regResult

# Core implementation of ICP
def icp_core(points1, points2):
    """
    solve transformation from points1 to points2, points of the same index are well matched
    :param points1: numpy array of points1, size = nx3, n is num of point
    :param points2: numpy array of points2, size = nx3, n is num of point
    :return: transformation matrix T, size = 4x4
    """
    assert points1.shape == points2.shape, 'point cloud size not match'

    T = np.zeros(shape=(4, 4))
    T[0:3, 0:3] = np.eye(3)
    T[3, 3] = 1

    # Step 1: calculate centroid
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # Step 2: de-centroid of points1 and points2
    points1_centered = points1 - centroid1
    points2_centered = points2 - centroid2

    # Step 3: compute H, which is sum of p1i'*p2i'^T
    H = np.dot(points1_centered.T, points2_centered)

    # Step 4: SVD of H (can use 3rd-party lib), solve R and t
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid2.T - np.dot(R, centroid1.T)

    # Step 5, combine R and t into transformation matrix T
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

def svd_based_icp_matched(pcd1, pcd2, method = 'none'):
    ''' Execute SVD-based ICP on matched point clouds
    
    Parameters
    ----------
    
    `pcd1` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): source point cloud
    `pcd2` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): target point cloud
    `method` (`string`): the method used. Options: 'none', 'downSampling'

    Returns
    -------
    `processing_time` (`float`): the processing time

    and Visualization of the matched point cloud
    '''
    if method == 'none':
        points1 = np.array(pcd1.points)
        points2 = np.array(pcd2.points)
        pass
    elif method == 'downSampling':
        points1_i = np.array(pcd1.points)
        points2_i = np.array(pcd2.points)
        points1_v3v, _ = preprocess_point_cloud(pcd1, 0.1) # Voxel size
        points2_v3v, _ = preprocess_point_cloud(pcd2, 0.1)
        print(f"After downsampling, the shape is: pcd1: {np.shape(np.array(points1_v3v.points))}, pcd2: {np.shape(np.array(points2_v3v.points))}")
        maxLen = min(len(np.array(points1_v3v.points)),len(np.array(points2_v3v.points)))
        # Downsampled pointcloud
        points1 = np.array(points1_v3v.points)[:maxLen]
        points2 = np.array(points2_v3v.points)[:maxLen]
    else:
        print("Invalid method. Valid method: 'none', 'downsampling'")
    start_time = time.time()
    T = icp_core(points1, points2)
    processing_time = time.time() - start_time
    print('------------transformation matrix------------')
    print(T)

    # # Todo: calculate transformed point cloud 1 based on T solved above, and name it pcd1_transformed (same format as point1)

    # calculate transformed point cloud 1 based on T solved above, and name it pcd1_transformed (same format as point1)
    pcd1_transformed = (T[:3,:3] @ points1.T + T[:3,3].reshape(-1,1)).T
    if method == 'downSampling':
        pcd1_transformed_i = (T[:3,:3] @ points1_i.T + T[:3,3].reshape(-1,1)).T
    # visualization
    if method == 'downSampling':
        mean_distance = mean_dist(pcd1_transformed_i, points2_i)
    else:
        mean_distance = mean_dist(pcd1_transformed, points2)
    print('mean_error= ' + str(mean_distance))
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd1_tran = o3d.geometry.PointCloud()
    pcd1_tran.points = o3d.utility.Vector3dVector(pcd1_transformed)
    pcd1.paint_uniform_color([1, 0, 0])
    pcd2.paint_uniform_color([0, 1, 0])
    pcd1_tran.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd1_tran, axis_pcd])
    return processing_time

def svd_based_icp_unmatched(pcd1, pcd2, n_iter, threshold, method= 'none'):
    ''' Execute SVD-based ICP on unmatched point clouds
    
    Parameters
    ----------
    
    `pcd1` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): source point cloud
    `pcd2` (`<class 'open3d.cuda.pybind.geometry.PointCloud'>`): target point cloud
    `n_iter` (`int`): the number of iterations. Being multiples of 5 makes the batch mean error calculated correctly
    `method` (`string`): the method used. Options: 'none', 'downSampling', 'globalReg', 'combined'

    Returns
    -------
    `processing_time` (`float`): the processing time
    `batch_meanDistance_list` (`list`): the list of batch mean errors
    and Visualization of the matched point cloud
    '''
    start_time = time.time()
    points1_i = np.array(pcd1.points) # Used for calculating mean errors
    points2_i = np.array(pcd2.points)
    if method == 'none':
        points1 = points1_i.copy() # Used for ICP
        points2 = points2_i.copy()
        T_ini = np.eye(4,4)
    elif method == 'downSampling':
        points1_v3v, points1_fpfh = preprocess_point_cloud(pcd1, 0.1) # Voxel size
        points2_v3v, points2_fpfh = preprocess_point_cloud(pcd2, 0.1)
        print(f"After downsampling, the shape is: pcd1: {np.shape(np.array(points1_v3v.points))}, pcd2: {np.shape(np.array(points2_v3v.points))}")
        maxLen = min(len(np.array(points1_v3v.points)),len(np.array(points2_v3v.points)))
        # Downsampled pointcloud
        points1 = np.array(points1_v3v.points)[:maxLen]
        points2 = np.array(points2_v3v.points)[:maxLen]
        T_ini = np.eye(4,4)
    elif method == 'globalReg':
        points1 = points1_i.copy()
        points2 = points2_i.copy()
        points1_v3v, points1_fpfh = preprocess_point_cloud(pcd1, 0.1) # Voxel size
        points2_v3v, points2_fpfh = preprocess_point_cloud(pcd2, 0.1)
        T_ini = execute_global_registration(points1_v3v, points2_v3v, points1_fpfh, points2_fpfh, 0.05).transformation
    elif method == 'combined':
        points1_v3v, points1_fpfh = preprocess_point_cloud(pcd1, 0.1) # Voxel size
        points2_v3v, points2_fpfh = preprocess_point_cloud(pcd2, 0.1)
        print(f"After downsampling, the shape is: pcd1: {np.shape(np.array(points1_v3v.points))}, pcd2: {np.shape(np.array(points2_v3v.points))}")
        maxLen = min(len(np.array(points1_v3v.points)),len(np.array(points2_v3v.points)))
        # Downsampled pointcloud for calculation
        points1 = np.array(points1_v3v.points)[:maxLen]
        points2 = np.array(points2_v3v.points)[:maxLen]
        T_ini = execute_global_registration(points1_v3v, points2_v3v, points1_fpfh, points2_fpfh, 0.05).transformation
    else:
        print("Invalid method. Valid method: 'none', 'downSampling', 'globalReg', 'combined'")
    T_accumulated = T_ini
    print('Initial T = ')
    print(T_ini)
    points1 = (T_ini[:3,:3] @ points1.T + T_ini[:3,3].reshape(-1,1)).T
    if method == 'globalReg' or method == 'combined':
        points1_i = (T_ini[:3,:3] @ points1_i.T + T_ini[:3,3].reshape(-1,1)).T
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 0, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis_pcd)
    vis.add_geometry(pcd2)

    start_time = time.time()
    batch_meanDistance = 0
    batch_meanDistance_list = []
    j = 0
    for i in range(n_iter):
        # find the nearest neighbors of each point in points1
        points2_nearest = []
        for p1 in points1:
            distances = np.linalg.norm(points2 - p1, axis=1)
            min_idx = np.argmin(distances)
            points2_nearest.append(points2[min_idx])
        points2_nearest = np.asarray(points2_nearest)

        # solve icp
        T = icp_core(points1, points2_nearest)

        # update accumulated T
        T_accumulated = T @ T_accumulated

        # update points1
        points1 = (T[:3,:3] @ points1.T + T[:3,3].reshape(-1,1)).T
        if method == 'downSampling' or method == 'combined':
            points1_i = (T[:3,:3] @ points1_i.T + T[:3,3].reshape(-1,1)).T
        print('-----------------------------------------')
        print('iteration = ' + str(i+1))
        print('accumulated T = ')
        print(T_accumulated)
        if method == 'downSampling' or method == 'combined':
            mean_distance = mean_dist(points1_i, points2_i)
        else:
            mean_distance = mean_dist(points1, points2)
        batch_meanDistance += mean_distance
        print('mean_error= ' + str(mean_distance))
        if (i+1)%5 == 0:
            j+=1
            print(f'Batch {j}\'s mean error:{batch_meanDistance/5}')
            batch_meanDistance_list.append(batch_meanDistance/5)
            batch_meanDistance = 0

        # visualization
        pcd1_transed = o3d.geometry.PointCloud()
        pcd1_transed.points = o3d.utility.Vector3dVector(points1)
        pcd1_transed.paint_uniform_color([1, 0, 0])
        vis.add_geometry(pcd1_transed)
        vis.poll_events()
        vis.update_renderer()
        vis.remove_geometry(pcd1_transed)

        if mean_distance < 0.00001 or mean_distance < threshold:
            print('Fully converged!')
            break
    processing_time = time.time() - start_time
    o3d.visualization.draw_geometries([axis_pcd, pcd2, pcd1_transed])
    return processing_time, batch_meanDistance_list

def mean_dist(points1, points2):
    ''' Calculate the mean distance of two list of points
    
    Parameters
    ----------
    
    `points1` (`ndarray`): point list 1
    `points2` (`ndarray`): point list 2

    Returns
    -------
    `dis_array` (`ndarray`): the mean distance of two list of points
    '''
    dis_array = []
    for i in range(points1.shape[0]):
        dif = points1[i] - points2[i]
        dis = np.linalg.norm(dif)
        dis_array.append(dis)
    dis_array = np.mean(np.array(dis_array))
    return dis_array

def main():
    pcd1 = o3d.io.read_point_cloud('data/bunny1.ply')
    pcd2 = o3d.io.read_point_cloud('data/bunny2.ply')
    # Change task here
    # task = 'task2' # 'task1', 'task2'
    if task == 'task1':

        # Change method here
        # method = 'none' # 'none', 'downsampling'

        processing_time = svd_based_icp_matched(pcd1, pcd2, method=method)
        print(f"Processing time of ICP with method '{method}': {processing_time}")

    elif task == 'task2':

        # Change method here
        # method = 'none' # 'none', 'downSampling', 'globalReg', 'combined'

        processing_time, meanError_list = svd_based_icp_unmatched(pcd1, pcd2, n_iter=60, threshold=0.05, method=method)
        print(f"Processing time of unmatched ICP with method '{method}': {processing_time}")
        for i in range(len(meanError_list)):
            print(f'Batch {i+1}\'s mean error:{meanError_list[i]}')
    else:
        print("Invalid task. Valid task: 'task1', 'task2'")

if __name__ == '__main__':
    main()
