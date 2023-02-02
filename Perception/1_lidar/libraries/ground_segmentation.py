import random
import numpy as np
import math

def ground_segmentation(points, method='brutal'):
    ''' Segment the ground points from the point cloud
    
    Parameters
    ----------
    
    `points` (`numpy.ndarray`): point cloud to be segmented 
      
    `method` (`string`): method used. Options: `'RANSAC'`, `'brutal'`

    Returns
    -------
    `ground_cloud` (`numpy.ndarray`): ground point cloud
    `segmented_cloud` (`numpy.ndarray`): segmented point cloud without ground points
    `idx_ground` (`list`): index of round point cloud
    `idx_segmented` (`list`): index of ground point cloud
    '''
    if method == 'RANSAC':
        # 屏蔽开始
        #初始化数据
        idx_segmented = []
        segmented_cloud = []
        iters = 100   #最大迭代次数  000002.bin：10
        sigma = 0.4     #数据和模型之间可接受的最大差值   000002.bin：0.5   000001.bin: 0.2  000000.bin: 0.15  002979.bin：0.15  004443.bin：0.4
        ##最好模型的参数估计和内点数目,平面表达方程为   aX + bY + cZ +D= 0
        best_a = 0
        best_b = 0
        best_c = 0
        best_d = 0
        pretotal = 0 #上一次inline的点数
        #希望的到正确模型的概率
        P = 0.99
        n = len(points)    #点的数目
        outline_ratio = 0.6   #e :outline_ratio   000002.bin：0.6    000001.bin: 0.5  000000.bin: 0.6   002979.bin：0.6
        for i in range(iters):
            ground_cloud = []
            idx_ground = []
            #step1 选择可以估计出模型的最小数据集，对于平面拟合来说，就是三个点
            sample_index = random.sample(range(n),3)    #重数据集中随机选取3个点
            point1 = points[sample_index[0]]
            point2 = points[sample_index[1]]
            point3 = points[sample_index[2]]
            #step2 求解模型
            ##先求解法向量
            point1_2 = (point1-point2)      #向量 poin1 -> point2
            point1_3 = (point1-point3)      #向量 poin1 -> point3
            N = np.cross(point1_3,point1_2)            #向量叉乘求解 平面法向量
            ##slove model 求解模型的a,b,c,d
            a = N[0]
            b = N[1]
            c = N[2]
            d = -N.dot(point1)
            #step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
            total_inlier = 0
            pointn_1 = (points - point1)    #sample（三点）外的点 与 sample内的三点其中一点 所构成的向量
            distance = abs(pointn_1.dot(N))/ np.linalg.norm(N)     #求距离
            ##使用距离判断inline
            idx_ground = (distance <= sigma)
            total_inlier = np.sum(idx_ground == True)    #统计inline得点数
            ##判断当前的模型是否比之前估算的模型
            if total_inlier > pretotal:                                           #     log(1 - p)
                iters = math.log(1 - P) / math.log(1 - pow(total_inlier / n, 3))  #N = ------------
                pretotal = total_inlier                                               #log(1-[(1-e)**s])
                #获取最好得 abcd 模型参数
                best_a = a
                best_b = b
                best_c = c
                best_d = d

            # 判断是否当前模型已经符合超过 inline_ratio
            if total_inlier > n*(1-outline_ratio):
                break
        print("iters = %f" %iters)
        #提取分割后得点
        idx_segmented = np.logical_not(idx_ground)
        ground_cloud = points[idx_ground]
        segmented_cloud = points[idx_segmented]
        return ground_cloud,segmented_cloud,idx_ground,idx_segmented

    if method == 'brutal':
        idx_ground = []
        idx_segmented = []
        for i in range(len(points)):
            if(points[i][2] >= 0.00000001):
                idx_segmented.append(i)
            else:
                idx_ground.append(i)
        ground_cloud = points[idx_ground]
        segmented_cloud = points[idx_segmented]
        return ground_cloud,segmented_cloud,idx_ground,idx_segmented