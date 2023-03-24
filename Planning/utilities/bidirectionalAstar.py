import matplotlib.pyplot as plt
import time
from utilities.Astar import AstarPlanner

class bidirectionalAstarPlanner(AstarPlanner):
    def __init__(self, map, connectivity, visualization):
        super().__init__(map, connectivity=connectivity, visualization=visualization)
        self.visited_forward = None
        self.visited_backward = None

    def _merge_paths(self, current_forward, concatenated_node):
        """
        Merge the paths found from the two searches.

        Args:
            current_forward (node): Latest node from the forward search.
            concatenated_node (node): Node to be concatenated.

        Returns:
            list: The combined path from start to goal.
            goal: the state of the goal.
        """
        path = [current_forward.state]
        while current_forward.parent:
            current_forward = current_forward.parent
            path.insert(0, current_forward.state)
        current_backward = concatenated_node
        while current_backward.parent:
            current_backward = current_backward.parent
            path.append(current_backward.state)
        return path

    def plan(self, start, end):
        start_node = self.node(start, 0, self.manhatten_distance(start, end), None)
        end_node = self.node(end, 0, self.manhatten_distance(end, start), None)
        self.visited_forward = set()
        self.visited_backward = set()
        self.visitedNode_backward = []
        open_list_forward = [start_node]
        open_list_backward = [end_node]
        if self.visualization:
            plt.imshow(self.grid_map, cmap='gray')
            plt.plot(start[0], start[1], 'ro')
            plt.plot(end[0], end[1], 'go')
            plt.axis('off')
            plt.tight_layout()
        start_time = time.time()
        while open_list_forward and open_list_backward:
            current_forward = min(open_list_forward, key=lambda x: x.g_cost + x.h_cost)
            current_backward = min(open_list_backward, key=lambda x: x.g_cost + x.h_cost)
            for state in self.visited_backward:
                if current_forward.state == state:
                    # path found
                    print(f"Path found between {start} and {end}.")
                    for n in self.visitedNode_backward:
                        if n.state == state:
                            node = n
                    if node.parent is None: # Forwarding directly to the end
                        path = [current_forward.state]
                        while current_forward.parent:
                            path.append(current_forward.parent.state)
                            current_forward = current_forward.parent
                        path.reverse()
                        return path, self.visited_forward.union(self.visited_backward), time.time() - start_time
                    
                    if node.state == start: # Backwarding directly to the start
                        path = [current_backward.state]
                        while current_backward.parent:
                            path.append(current_backward.parent.state)
                            current_backward = current_backward.parent
                        path.reverse()
                        return path, self.visited_forward.union(self.visited_backward), time.time() - start_time
                    
                    concatenated_node = node.parent
                    path = self._merge_paths(current_forward, concatenated_node)
                    return path, self.visited_forward.union(self.visited_backward), time.time()-start_time
                
            open_list_forward.remove(current_forward)
            open_list_backward.remove(current_backward)
            self.visited_forward.add(current_forward.state)
            self.visited_backward.add(current_backward.state)
            self.visitedNode_backward.append(current_backward)
            if self.visualization:
                plt.plot(current_forward.state[0], current_forward.state[1], "ro")
                plt.plot(current_backward.state[0], current_backward.state[1], "ro")

            neighbors_f = [(current_forward.state[0] + dx, current_forward.state[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
            neighbors_b = [(current_backward.state[0] + dx, current_backward.state[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

            if self.connectivity == 4:
                neighbors_f = [(x, y) for x, y in neighbors_f if abs(x - current_forward.state[0]) + abs(y - current_forward.state[1]) <= 1]
                neighbors_b = [(x, y) for x, y in neighbors_b if abs(x - current_backward.state[0]) + abs(y - current_backward.state[1]) <= 1]

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

            for neighbor in neighbors_b:
                if self.is_valid_point(neighbor) and neighbor not in self.visited_backward:
                    g = current_backward.g_cost + self.calculate_g(neighbor, current_backward.state)
                    in_open_list = False
                    for node_in_list in open_list_backward:
                        if node_in_list.state == neighbor:
                            in_open_list = True
                            if g < node_in_list.g_cost:
                                node_in_list.g_cost = g
                                node_in_list.parent = current_backward
                            break
                    if not in_open_list:
                        h = self.calculate_h(neighbor, start)
                        neighbor_node = self.node(neighbor, g, h, current_backward)
                        open_list_backward.append(neighbor_node)

                        if self.visualization:
                            plt.plot(neighbor_node.state[0], neighbor_node.state[1], "bo")
            if time.time()-start_time >= 90:
                print(f"Run out of time, no path between {start} and {end} found.")
                break
            if self.visualization:
                plt.pause(0.001)
        return None, self.visited_forward.union(self.visited_backward), 0