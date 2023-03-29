import numpy as np
import matplotlib.pyplot as plt
import itertools

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
    # Initialize variables to keep track of success rate, paths, visited cells, time, distances, and count
    success = 0
    paths = []
    visited_cells_list = []
    total_time = 0
    distances_total = []
    count = 0
    
    # Loop through all pairs of points in the list
    for i in range(len(points)):
        distances = []
        for j in range(len(points)):
            # If the start and end points are the same, the distance is 0
            if i == j:
                distances.append(0)
                continue
            visited_cells = set()
            # Plan a path from the current start and end points using the provided planner object
            path, visited, time = planner.plan(points[i], points[j])
            # If a path is found, update the success rate and add the path to the list of paths
            if path is not None:
                success += 1
                # Calculate the length of the path and add it to the distances list
                path_length = calc_pathLength(path, planner)
                distances.append(round(path_length, 2))
            # If no path is found, add a dash to the distances list to indicate failure
            else:
                count -= 1
                distances.append('-')

            # Update the total time and count variables
            total_time += time
            paths.append(path)
            visited_cells.update(visited)
            count += 1
        visited_cells_list.append(visited_cells)
        distances_total.append(distances)

    # Calculate the average time and average length of successful paths
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

    # Calculate the success rate and return the paths, distances, visited cells, average time, average length, and success rate
    success_rate = success/(len(points)**2-len(points))
    return paths, distances_total, visited_cells_list, avg_time, average_length, success_rate





def visPath(locations_dict, paths, grid_map):
    # Visualize the map and paths
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid_map, cmap='gray')

    for key, value in locations_dict.items():
        plt.plot(locations_dict[key][0], locations_dict[key][1], marker="o", markersize=10, markeredgecolor="red")
        plt.text(locations_dict[key][0], locations_dict[key][1]-15, s=key, fontsize='x-large', fontweight='bold', color='b', ha='center')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, path in enumerate(paths):
        if path is not None:
            color=colors[i % len(colors)]
            # extract x and y coordinates
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            # calculate arrow directions
            dx = [path[i+1][0] - path[i][0] for i in range(len(path)-1)]
            dy = [path[i+1][1] - path[i][1] for i in range(len(path)-1)]
            ax.quiver(x_coords[:-1:20], y_coords[:-1:20], dx[::20], dy[::20], angles='xy', scale_units='xy', width = 0.009, scale=1.2, color = color, zorder = 10)
            path_array = np.array(path)
            ax.plot(path_array[:,0], path_array[:,1], color=color, linewidth=2, alpha=0.7)

    ax.set_title('Occupancy Grid Map')
    plt.show()


def nearest_neighbor(dist, visited):
    min_dist = float('inf')
    next_node = -1
    for i in range(dist.shape[0]):
        if i not in visited and dist[visited[-1], i] < min_dist:
            min_dist = dist[visited[-1], i]
            next_node = i
    return next_node


def TSP(paths, distances, method='nearestNeighbor'):
    # Starting point
    start = 0

    if method == 'nearestNeighbor':
        # Visited nodes
        visited = [start]

        # Find nearest neighbor until all nodes are visited
        while len(visited) < distances.shape[0]:
            next_node = nearest_neighbor(distances, visited)
            visited.append(next_node)

        # Add starting point to the end to complete the loop
        visited.append(start)

        # Print the optimal route and total distance
        print("Optimal route: ", visited)

        # Define a dictionary to map integers to store names
        store_names = {1: 'Snacks', 2: 'Store', 3: 'Movie', 4: 'Food', 0: 'Start'}

        # Process the optimal route to replace integers with store names
        optimal_route_names = [store_names[i] for i in visited]

        # Print the final shortest route on the map, and the total distance
        print(" -> ".join(optimal_route_names) + "")
        print("Total distance: ", sum([distances[visited[i], visited[i+1]] for i in range(len(visited)-1)]))

        path2d = np.zeros((5, 5)).tolist()
        index = 0
        for i in range(5):
            for j in range(5):
                if i != j:
                    path2d[i][j] = paths[index]
                    index += 1
        optimalRoute = []
        for i in range(len(visited)):
            if i != len(visited)-1:
                optimalRoute.append(path2d[visited[i]][visited[i+1]])
        return optimalRoute
    
    elif method == 'bruteForce':
        num_stores = distances.shape[0] - 1 # Excluding the start point
        store_indices = list(range(1, num_stores+1)) # Indices of stores in the distances matrix

        # Generate all possible permutations of store indices
        permutations = list(itertools.permutations(store_indices))

        # Add start and end indices to each permutation to complete the loop
        all_routes = [[0] + list(p) + [0] for p in permutations]

        # Calculate the total distance for each route
        route_distances = [sum([distances[r[i], r[i+1]] for i in range(len(r)-1)]) for r in all_routes]

        # Find the index of the route with the minimum distance
        min_distance_index = route_distances.index(min(route_distances))

        # Get the optimal route
        optimalRoute = all_routes[min_distance_index]

        # Print the optimal route and total distance
        print("Optimal route: ", optimalRoute)

        # Define a dictionary to map integers to store names
        store_names = {1: 'Snacks', 2: 'Store', 3: 'Movie', 4: 'Food', 0: 'Start'}

        # Process the optimal route to replace integers with store names
        optimal_route_names = [store_names[i] for i in optimalRoute]

        # Print the final shortest route on the map, and the total distance
        print(" -> ".join(optimal_route_names) + "")
        print("Total distance: ", min(route_distances))

        path2d = np.zeros((5, 5)).tolist()
        index = 0
        for i in range(5):
            for j in range(5):
                if i != j:
                    path2d[i][j] = paths[index]
                    index += 1
        optimalRoute2d = []
        for i in range(len(optimalRoute)):
            if i != len(optimalRoute)-1:
                optimalRoute2d.append(path2d[optimalRoute[i]][optimalRoute[i+1]])
        return optimalRoute2d