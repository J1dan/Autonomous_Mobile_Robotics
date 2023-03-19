import numpy as np


def inflate_map(grid_map):
    kernel = np.zeros((3, 3))  # 3x3 kernel for 8-connectivity
    inflated_map = np.copy(grid_map)  # create a copy of the map

    for i in range(1, len(grid_map)-1):
        for j in range(1, len(grid_map[0])-1):
            if grid_map[i, j] == 0:  # if the pixel is black
                inflated_map[i-1:i+2, j-1:j+2] = np.minimum(inflated_map[i-1:i+2, j-1:j+2], kernel)
    inflated_map[0] = 0
    inflated_map[:,0] = 0
    inflated_map[len(grid_map)-1] = 0
    inflated_map[:,len(grid_map)-1] = 0
    return inflated_map

def calc_pathLength(path, planner, resolution=0.2):
    path_length = 0
    for n in range(len(path)):
        if n == len(path)-1:
            break
        path_length += planner.euclidean_distance(path[n], path[n+1]) * resolution
    return path_length

def plan_all_paths(points, planner):
    success = 0
    paths = []
    visited_cells_list = []
    total_time = 0
    distances_total = []
    count = 0
    for i in range(len(points)):
        distances = []
        for j in range(len(points)):
            if i == j:
                distances.append(0)
                continue
            visited_cells = set()
            path, visited, time = planner.plan(points[i], points[j])
            if path is not None:
                success += 1
                path_length = calc_pathLength(path, planner)
                distances.append(round(path_length, 2))
            else:
                count -= 1
                distances.append('-')

            total_time += time
            paths.append(path)
            visited_cells.update(visited)
            count += 1
        visited_cells_list.append(visited_cells)
        distances_total.append(distances)

    avg_time = total_time/count
    total = 0
    c = 0
    for row in distances_total:
        for elem in row:
            if elem != '-' and elem != 0:
                total += elem
                c += 1
    if c == 0:
        average_length = 0
    else:
        average_length = total / c
    success_rate = success/(len(points)**2-len(points))
    return paths, distances_total, visited_cells_list, avg_time, average_length, success_rate