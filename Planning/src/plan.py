import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utilities.Astar import AstarPlanner
from utilities.hybridAstar_upgraded import hybridAstarPlanner
from utilities.Dikstra import DijkstraPlanner
from utilities.greedyBest import greedyPlanner
from utilities.bidirectionalAstar import bidirectionalAstarPlanner
from utilities.switchAstar import switchAstarPlanner
from utilities.utility import inflate_map
from utilities.utility import TSP
from utilities.utility import visPath
from utilities.utility import plan_all_paths
import imageio
import matplotlib.pyplot as plt

# Locations on the map
locations = {'start': (345, 95),    # Start from the level 2 Escalator
             'snacks': (470, 475),  # Garrett Popcorn
             'store': (20, 705),    # DJI Store
             'movie': (940, 545),   # Golden Village
             'food': (535, 800),    # PUTIEN
            }
start = locations['start']
snacks = locations['snacks']
store = locations['store']
movie = locations['movie']
food = locations['food']

points = [start, snacks, store, movie, food]
# points = [start, store]

# Load the occupancy grid map
grid_map = imageio.imread('./Planning/map/vivocity_freespace.png')

# Preprocess the map
inflated_map = inflate_map(grid_map)

planner = AstarPlanner(inflated_map, connectivity=8, visualization=False)

paths, distances_total, visited_cells_list, avg_time, avg_distance, success_rate = plan_all_paths(points, planner)
print(f"distance list: {distances_total}")
print(f"Avg time consumed: {avg_time}")
print(f"Avg length of path: {avg_distance}")
print(f"Success rate: {100*round(success_rate, 2)}%")
# Visualize the map and paths
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid_map, cmap='gray')

for i, end in enumerate(points):
    ax.scatter(end[0], end[1], c='green', s=50, marker='o')

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i, path in enumerate(paths):
    if path is not None:
        path_array = np.array(path)
        ax.plot(path_array[:,0], path_array[:,1], color=colors[i % len(colors)], linewidth=2, alpha=0.7)

# Traveling Seller Problem

# Distance matrix
distances = np.array(distances_total)

optimalRoute = TSP(paths, distances, 'nearestNeighbor') # Options: 'bruteForce', 'nearestNeighbor'

visPath(locations, optimalRoute, grid_map)
