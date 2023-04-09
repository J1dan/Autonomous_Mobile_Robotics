import numpy as np
from math import pi
from utilities.Astar import AstarPlanner
import matplotlib.pyplot as plt
import time

import numpy as np
from math import pi

class hybridAstarPlanner(AstarPlanner):
    def __init__(self, map, connectivity=None, resolution=0.2, max_steering=pi/4, num_steps=20, visualization=False):
        # Call the constructor of the base class (AstarPlanner)
        super().__init__(map, resolution, visualization=visualization)
        # Set the weight for curvature-based cost
        self.curvature_weight = 0.2
        # Set the maximum steering angle for the car
        self.max_steering = max_steering
        # Set the number of steps for the trajectory
        self.num_steps = num_steps
    
    class hybrid_node(object):
        def __init__(self, state, x_list, y_list, yaw_list, g_cost, h_cost, parent):
            # Store the state of the hybrid node (x, y, theta)
            self.state = state
            # Store the list of x, y, and yaw values for the trajectory
            self.x_list = x_list
            self.y_list = y_list
            self.yaw_list = yaw_list
            # Store the g-cost and h-cost values of the hybrid node
            self.g_cost = g_cost
            self.h_cost = h_cost
            # Store a reference to the parent hybrid node
            self.parent = parent

    def calculate_h(self, current, goal):
        # Use Euclidean distance as the heuristic for A*
        return self.euclidean_distance(current, goal)

    def generate_trajectory(self, x, y, theta, steer, v):
        # Calculate the time step based on the current velocity
        dt = 0.3 / v
        # Initialize the trajectory with the current state
        traj = [(x, y, theta)]
        # Initialize the length of the trajectory to zero
        traj_length = 0
        # Generate a trajectory with a fixed number of steps
        for i in range(self.num_steps):
            # Stop generating the trajectory if the length is greater than 1.4 meters
            if traj_length > 1.4:
                break
            # Update the angle based on the steering angle and time step
            theta += steer * dt
            # Update the x and y positions based on the angle, velocity, and time step
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            # Update the length of the trajectory
            traj_length += v * dt
            # Add the new state to the trajectory
            traj.append([x, y, theta])
        # Return the trajectory
        return traj

    def is_valid_hybrid_node(self, x, y):
        # Check if the point (x, y) is valid (i.e., not occupied by an obstacle)
        return self.is_valid_point((int(round(x)), int(round(y))))

    def car_cost(self, p1, p2):
        # Unpack x, y, and theta from each point
        x1, y1, theta1 = p1
        x2, y2, theta2 = p2
        # Calculate steering angle
        steer = np.arctan2(y2 - y1, x2 - x1) - theta1
        # Generate trajectory based on steering angle and velocity
        traj = self.generate_trajectory(x1, y1, theta1, steer, 1.0)
        # Initialize cost to zero
        cost = 0
        # Iterate over each segment in the trajectory
        for i in range(len(traj) - 2):
            # Get points for current segment, next segment, and segment after that
            p1 = traj[i]
            p2 = traj[i + 1]
            p3 = traj[i + 2]
            # Calculate curvature using finite differences
            dx1 = p1[0] - p2[0]
            dy1 = p1[1] - p2[1]
            dx2 = p2[0] - p3[0]
            dy2 = p2[1] - p3[1]
            if abs(dx1) < 1e-6 and abs(dy1) < 1e-6:
                k1 = 0
            else:
                k1 = 2 * (dx1 * dy2 - dy1 * dx2) / (dx1 ** 2 + dy1 ** 2) ** 1.5
            # Add the distance between the current and next point and the curvature-based cost to the total cost
            cost += 0.2*self.neighboring_distance((int(round(p1[0])), int(round(p1[1]))),
                                            (int(round(p2[0])), int(round(p2[1])))) + abs(k1) * self.curvature_weight
        # Return the total cost
        return cost


    def plan(self, start, end):
        # Initialize the visited set
        self.visited = set()

        # Calculate the initial orientation
        initial_orientation = np.arctan2(end[1] - start[1], end[0] - start[0])

        # Create the start node
        start_node = self.hybrid_node((start[0], start[1], int(round(initial_orientation / (pi/12)))*pi/12), [start[0]], [start[1]], [initial_orientation], 0, self.manhatten_distance(start, end), None)

        # Add the start node to the open list
        open_list = [start_node]

        # If visualization is enabled, plot the start and end points on the grid map
        if self.visualization:
            plt.imshow(self.grid_map, cmap='gray')
            plt.plot(start[0], start[1], 'ro', zorder=100)
            plt.plot(end[0], end[1], 'go')
            plt.axis('off')
            plt.tight_layout()

        # Record the starting time
        start_time = time.time()

        # Loop until the open list is empty
        while open_list:
            # Select the node with the lowest f-cost from the open list
            current = min(open_list, key=lambda x: x.g_cost + x.h_cost)

            # Remove the current node from the open list
            open_list.remove(current)

            # If the current node is at the goal, return the path
            if current.state[0:2] == end:
                print(f"Path found between {start} and {end}.")
                path = [end]

                # Follow the parent pointers to build the path
                while current.parent:
                    path.append(current.parent.state[0:2])
                    current = current.parent
                path.reverse()
                return path, self.visited, time.time()-start_time

            # Add the current node to the visited set
            self.visited.add(current)

            # If visualization is enabled, plot the current node on the grid map
            if self.visualization:
                plt.plot(current.state[0], current.state[1], "ro", zorder=100)

            # Generate trajectories for each possible steering angle
            for steer in [0, -self.max_steering*0.5, self.max_steering*0.5, -self.max_steering, self.max_steering]:
                traj = self.generate_trajectory(current.x_list[-1], current.y_list[-1], current.yaw_list[-1], steer, 1.0)

                # Loop through the states along the trajectory
                for i in range(len(traj)):
                    # Create the next state
                    next_state = (int(round(traj[i][0])), int(round(traj[i][1])), int(round(traj[i][2] / (pi/12)))*pi/12)

                    # Initialize the flag variable to check if the next node is already in the open list or visited set
                    FLAG = False

                    # If the next state is not valid, skip it
                    if not self.is_valid_hybrid_node(next_state[0], next_state[1]):
                        continue

                    # Calculate the g-cost and h-cost of the next node
                    next_g_cost = current.g_cost + self.car_cost(traj[0], traj[i])
                    next_h_cost = self.calculate_h(next_state[0:2], end)

                    # Create the next node
                    next_node = self.hybrid_node(next_state, np.array(traj)[:i,0], np.array(traj)[:i,1], np.array(traj)[:,2], next_g_cost, next_h_cost, current)
                    
                    # Check if already exist in open_list to accelerate
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