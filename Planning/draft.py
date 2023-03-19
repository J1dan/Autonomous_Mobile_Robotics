import numpy as np
from queue import PriorityQueue
q = PriorityQueue()
q.put(1)
q.put(2)

while not q.empty():
    element = q.get()
    print(element)