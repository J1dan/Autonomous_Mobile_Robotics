from utilities.Astar import AstarPlanner

class DijkstraPlanner(AstarPlanner):
    def __init__(self, map, connectivity, visualization):
        super().__init__(map, connectivity=connectivity, visualization=visualization)

    def calculate_h(self, current, goal):
        return 0