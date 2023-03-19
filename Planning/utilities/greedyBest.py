from utilities.Astar import AstarPlanner

class greedyPlanner(AstarPlanner):
    def __init__(self, map, connectivity, visualization):
        super().__init__(map, connectivity=connectivity, visualization=visualization)

    def calculate_g(self, p1, p2):
        return 0