import numpy as np
import imageio
import matplotlib.pyplot as plt
import time

class AstarPlanner:
    def __init__(self, map, resolution=0.2, connectivity=8, visualization = False):
        self.grid_map = map
        self.resolution = resolution
        self.connectivity = connectivity
        self.visited = None
        self.visualization = visualization

    class node(object):
        def __init__(self, state, g_cost, h_cost, parent):
            self.state = state
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.parent = parent

    def manhatten_distance(self, p1, p2):
        return abs(p1[0]- p2[0]) + abs(p1[1]- p2[1])
    
    def calculate_g(self, p1, p2):
        return self.neighboring_distance(p1, p2)

    def calculate_h(self, current, goal):
        return self.manhatten_distance(current, goal)

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def neighboring_distance(self, p1, p2):
        if p1[0] == p2[0] or p1[1] == p2[1]:
            return 1
        else:
            return 1.414

    def is_valid_point(self, point):
        x, y = point
        return 0 <= x < self.grid_map.shape[1] and 0 <= y < self.grid_map.shape[0] and self.grid_map[y, x] == 255

    def plan(self, start, end):
        self.visited = set()
        start_node = self.node(start, 0, self.manhatten_distance(start, end), None)
        open_list = [start_node]
        if self.visualization:
            plt.imshow(self.grid_map, cmap='gray')
            plt.plot(start[0], start[1], 'ro')
            plt.plot(end[0], end[1], 'go')
            plt.axis('off')
            plt.tight_layout()
        start_time = time.time()
        while open_list:
            current = min(open_list, key=lambda x: x.g_cost + x.h_cost)
            open_list.remove(current)
            if current.state == end:
                print(f"Path found between {start} and {end}.")
                path = [end]
                goal = current
                while current.parent:
                    path.append(current.parent.state)
                    current = current.parent
                path.reverse()
                return path, self.visited, time.time() - start_time
            self.visited.add(current.state)
            if self.visualization:
                plt.plot(current.state[0], current.state[1], "ro")
            neighbors = [(current.state[0] + dx, current.state[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
            if self.connectivity == 4:
                neighbors = [(x, y) for x, y in neighbors if abs(x - current.state[0]) + abs(y - current.state[1]) <= 1]
            for neighbor in neighbors:
                if self.is_valid_point(neighbor) and neighbor not in self.visited:
                    g = current.g_cost + self.calculate_g(neighbor, current.state)
                    in_open_list = False
                    for node_in_list in open_list:
                        if node_in_list.state == neighbor:
                            in_open_list = True
                            if g < node_in_list.g_cost:
                                node_in_list.g_cost = g
                                node_in_list.parent = current
                            break
                    if not in_open_list:
                        h = self.calculate_h(neighbor, end)
                        neighbor_node = self.node(neighbor, g, h, current)
                        open_list.append(neighbor_node)
                        if self.visualization:
                            plt.plot(neighbor_node.state[0], neighbor_node.state[1], "bo")
            if self.visualization:
                plt.pause(0.001)
        return None, self.visited, 0