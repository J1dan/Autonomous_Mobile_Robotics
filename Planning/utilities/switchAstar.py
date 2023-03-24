from utilities.Astar import AstarPlanner

class switchAstarPlanner(AstarPlanner):
    def __init__(self, map, connectivity, visualization):
        super().__init__(map, connectivity=connectivity, visualization=visualization)

    def calc_pathLength(self, path):
        path_length = 0
        for n in range(len(path)):
            if n == len(path)-1:
                break
            path_length += self.neighboring_distance(path[n], path[n+1]) * self.resolution
        return path_length
    
    def plan(self, start, end):
        planner = AstarPlanner(self.grid_map, connectivity=self.connectivity, visualization=self.visualization)
        path, visited, time = planner.plan(start, end)
        path_reverse, visited_reverse, time_reverse = planner.plan(end, start)
        if path is not None:
            l_forward = self.calc_pathLength(path)
            l_reverse = self.calc_pathLength(path_reverse)
        if l_forward > l_reverse:
            return path_reverse, visited_reverse, time + time_reverse
        else:
            return path, visited, time + time_reverse
        