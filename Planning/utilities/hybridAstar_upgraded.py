import numpy as np
from math import pi
from utilities.Astar import AstarPlanner
import matplotlib.pyplot as plt
import time

import numpy as np
from math import pi

class hybridAstarPlanner(AstarPlanner):
    def __init__(self, map, connectivity=None, resolution=0.2, max_steering=pi/4, num_steps=20, visualization = False):
        super().__init__(map, resolution, visualization=visualization)
        self.curvature_weight = 0.2
        self.max_steering = max_steering
        self.num_steps = num_steps
    
    class hybrid_node(object):
        def __init__(self, state, x_list, y_list, yaw_list, g_cost, h_cost, parent):
            self.state = state
            self.x_list = x_list
            self.y_list = y_list
            self.yaw_list = yaw_list
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.parent = parent

    def calculate_h(self, current, goal):
        return self.euclidean_distance(current, goal)

    def generate_trajectory(self, x, y, theta, steer, v):
        dt = 0.3 / v
        traj = [(x, y, theta)]
        traj_length = 0
        for i in range(self.num_steps):
            if traj_length > 1.4:
                break
            theta += steer * dt
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            traj_length += v * dt
            traj.append([x, y, theta])
        return traj

    def is_valid_hybrid_node(self, x, y):
        return self.is_valid_point((int(round(x)), int(round(y))))

    def car_cost(self, p1, p2):
        x1, y1, theta1 = p1
        x2, y2, theta2 = p2
        steer = np.arctan2(y2 - y1, x2 - x1) - theta1
        traj = self.generate_trajectory(x1, y1, theta1, steer, 1.0)
        cost = 0
        for i in range(len(traj) - 2):
            p1 = traj[i]
            p2 = traj[i + 1]
            p3 = traj[i + 2]
            # calculate curvature
            dx1 = p1[0] - p2[0]
            dy1 = p1[1] - p2[1]
            dx2 = p2[0] - p3[0]
            dy2 = p2[1] - p3[1]
            if abs(dx1) < 1e-6 and abs(dy1) < 1e-6:
                k1 = 0
            else:
                k1 = 2 * (dx1 * dy2 - dy1 * dx2) / (dx1 ** 2 + dy1 ** 2) ** 1.5
            # add curvature-based cost
            cost += 0.2*self.neighboring_distance((int(round(p1[0])), int(round(p1[1]))),
                                            (int(round(p2[0])), int(round(p2[1])))) + abs(k1) * self.curvature_weight
        return cost

    def plan(self, start, end):
        self.visited = set()
        initial_orientation = np.arctan2(end[1] - start[1], end[0] - start[0])
        start_node = self.hybrid_node((start[0], start[1], int(round(initial_orientation / (pi/12)))*pi/12), [start[0]], [start[1]], [initial_orientation], 0, self.manhatten_distance(start, end), None)
        open_list = [start_node]
        if self.visualization:
            plt.imshow(self.grid_map, cmap='gray')
            plt.plot(start[0], start[1], 'ro', zorder=100)
            plt.plot(end[0], end[1], 'go')
            plt.axis('off')
            plt.tight_layout()
        start_time = time.time()
        while open_list:
            current = min(open_list, key=lambda x: x.g_cost + x.h_cost)
            open_list.remove(current)

            if current.state[0:2] == end:
                print(f"Path found between {start} and {end}.")
                path = [end]
                while current.parent:
                    path.append(current.parent.state[0:2])
                    current = current.parent
                path.reverse()
                return path, self.visited, time.time()-start_time
            self.visited.add(current)
            if self.visualization:
                plt.plot(current.state[0], current.state[1], "ro", zorder=100)
            for steer in [0, -self.max_steering*0.5, self.max_steering*0.5, -self.max_steering, self.max_steering]:
                traj = self.generate_trajectory(current.x_list[-1], current.y_list[-1], current.yaw_list[-1], steer, 1.0)
                for i in range(len(traj)):
                    next_state = (int(round(traj[i][0])), int(round(traj[i][1])), int(round(traj[i][2] / (pi/12)))*pi/12)
                    # next_state = (int(round(traj[-1][0])), int(round(traj[-1][1])), int(round(traj[-1][2] / (pi/12)))*pi/12)
                    FLAG = False

                    if not self.is_valid_hybrid_node(next_state[0], next_state[1]):
                        continue
                    next_g_cost = current.g_cost + self.car_cost(traj[0], traj[i])
                    next_h_cost = self.calculate_h(next_state[0:2], end)
                    next_node = self.hybrid_node(next_state, np.array(traj)[:i,0], np.array(traj)[:i,1], np.array(traj)[:,2], next_g_cost, next_h_cost, current)
                    for node in open_list:
                        if node.state[0:2] == next_state[0:2]:
                            FLAG = True
                            break
                    if FLAG:
                        continue

                    for node in self.visited:
                        if node.state[0:2] == next_state[0:2]:
                            FLAG = True
                            break
                    if FLAG:
                        continue
                    open_list.append(next_node)

                    if self.visualization:
                        plt.plot(next_node.state[0], next_node.state[1], "bo")
                    if self.visualization:
                        plt.plot(np.array(traj)[:,0], np.array(traj)[:,1], "yo", markersize=3)
            if self.visualization:
                plt.pause(0.001)
            if time.time()-start_time >= 90:
                print(f"Run out of time, no path between {start} and {end} found.")
                break
        return None, self.visited, 0


