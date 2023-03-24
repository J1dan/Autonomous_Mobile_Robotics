import matplotlib.pyplot as plt
import time
from utilities.Astar import AstarPlanner

class bidirectionalAstarPlanner(AstarPlanner):
    def __init__(self, map, connectivity, visualization):
        super().__init__(map, connectivity=connectivity, visualization=visualization)

    def plan(self, start, end):
        self.visited_forward = set()
        start_node = self.node(start, 0, self.manhatten_distance(start, end), None)
        open_list_forward = [start_node]
        if self.visualization:
            plt.imshow(self.grid_map, cmap='gray')
            plt.plot(start[0], start[1], 'ro')
            plt.plot(end[0], end[1], 'go')
            plt.axis('off')
            plt.tight_layout()
        start_time = time.time()
        while open_list_forward:
            current_forward = min(open_list_forward, key=lambda x: x.g_cost + x.h_cost)
            open_list_forward.remove(current_forward)
            if current_forward.state == end:
                print(f"Path found between {start} and {end}.")
                path = [end]
                while current_forward.parent:
                    path.append(current_forward.parent.state)
                    current_forward = current_forward.parent
                path.reverse()
                return path, self.visited_forward, time.time() - start_time
            self.visited_forward.add(current_forward.state)
            if self.visualization:
                plt.plot(current_forward.state[0], current_forward.state[1], "ro")
            neighbors_f = [(current_forward.state[0] + dx, current_forward.state[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
            # neighbors_b = [(current_backward.state[0] + dx, current_backward.state[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

            if self.connectivity == 4:
                neighbors_f = [(x, y) for x, y in neighbors_f if abs(x - current_forward.state[0]) + abs(y - current_forward.state[1]) <= 1]
            for neighbor in neighbors_f:
                if self.is_valid_point(neighbor) and neighbor not in self.visited_forward:
                    g = current_forward.g_cost + self.calculate_g(neighbor, current_forward.state)
                    in_open_list = False
                    for node_in_list in open_list_forward:
                        if node_in_list.state == neighbor:
                            in_open_list = True
                            if g < node_in_list.g_cost:
                                node_in_list.g_cost = g
                                node_in_list.parent = current_forward
                            break
                    if not in_open_list:
                        h = self.calculate_h(neighbor, end)
                        neighbor_node = self.node(neighbor, g, h, current_forward)
                        open_list_forward.append(neighbor_node)
                        if self.visualization:
                            plt.plot(neighbor_node.state[0], neighbor_node.state[1], "bo")
            if self.visualization:
                plt.pause(0.001)
        return None, self.visited_forward, 0