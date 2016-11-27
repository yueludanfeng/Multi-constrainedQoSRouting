# coding=utf-8
from random import randint
class Graph:
    def __init__(self, node_num, edge_num):
        self.node_num = node_num
        self.edge_num = edge_num
        self.bandwidth = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        self.delay = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        self.cost = [[0 for col in range(self.node_num)] for row in range(self.node_num)]

    def add_bandwidth(self, i, j, value = 0):
        self.bandwidth[i][j] = value

    def add_cost(self, i, j, value = 0):
        self.cost[i][j] = value

    def add_delay(self, i, j, value = 0):
        self.delay[i][j] = value

    def add_edge_measure(self, i, j, bandwidth = 0, delay = 0, cost = 0):
        self.add_bandwidth(i, j, bandwidth)
        self.add_delay(i, j, delay)
        self.add_cost(i, j, cost)


# 染色体(解)
class Population:
    # DFS
    def __init__(self, graph, src_code, des_code, pop_size):
        self.chromosomes = [[] for i in range(pop_size)]
        self.node_num = graph.node_num
        self.pop_size = pop_size
        for i in range(pop_size):
            for j in range(self.node_num):
                self.chromosomes[i].append(randint(0, 1))
            self.chromosomes[i][src_code] = 1
            self.chromosomes[i][des_code] = 1

    def print_chromosome(self):
        for i in range(self.pop_size):
            print self.chromosomes[i]
        # 根据des_code逆向找到一条路径即为一个解

    def calculateFitness(self):
        print




graph = Graph(5, 7)
graph.add_edge_measure(0, 1, 1, 2, 3)
graph.add_edge_measure(0, 2, 4, 5, 7)
graph.add_edge_measure(1, 3, 4, 5, 6)
graph.add_edge_measure(1, 4, 3, 4, 5)
graph.add_edge_measure(2, 4, 2, 3, 6)
graph.add_edge_measure(3, 4, 7, 8, 9)
graph.add_edge_measure(2, 3, 2, 4, 1)
print graph.bandwidth[0][1]
# print graph.cost[0][1]

pop = Population(graph, 0, 4, 4)
pop.print_chromosome()
