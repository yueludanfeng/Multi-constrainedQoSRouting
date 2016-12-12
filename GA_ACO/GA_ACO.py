# coding=utf-8
# Author:   Lixinming
# 融合了遗传算法和蚁群算法的混合算法
# 关键在于如何融合：先遗传，在蚁群，蚁群算法的信息素初始值如何由遗传算法获得
from random import choice

class Graph:
    def __init__(self, node_num, edge_num):
        self.node_num = node_num
        self.edge_num = edge_num
        self.connection_status = [[False for col in range(self.node_num)] for col in range(self.node_num)]
        self.band_width = [[0 for col in range(self.node_num)] for col in range(self.node_num)]
        self.delay = [[0 for col in range(self.node_num)] for col in range(self.node_num)]
        self.cost = [[0 for col in range(self.node_num)] for col in range(self.node_num)]
        self.node_adjs = {r: [] for r in range(self.node_num)}
        self.total_cost = 0
        self.total_delay = 0

    def set_connection_status(self, row, col):
        self.connection_status[row][col] = True
        self.connection_status[col][row] = True

    def get_connection_status(self):
        return self.connection_status

    def get_delay(self):
        return self.delay

    def get_cost(self):
        return self.cost

    def get_total_cost(self):
        return self.total_cost

    def get_total_delay(self):
        return self.total_delay

    def get_node_adjs(self):
        return self.node_adjs

    def set_band_width(self, row, col, value):
        self.band_width[row][col] = value
        self.band_width[col][row] = value

    def set_delay(self, row, col, value):
        self.delay[row][col] = value
        self.delay[col][row] = value
        self.total_delay += value

    def set_cost(self, row, col, value):
        self.cost[row][col] = value
        self.cost[col][row] = value
        self.total_cost += value



    def set_edge_measure(self, row, col, bandwidth, delay, cost):
        self.set_band_width(row, col, bandwidth)
        self.set_delay(row, col, delay)
        self.set_cost(row, col, cost)
        self.set_connection_status(row, col)

    # 初始化边上的各个度量
    # param: fp 指的是文件对象
    # param: length 指的是参数长度
    def init_edge_measure(self, fp, length):
        # 随着后期参数的增加，length的值也会增加
        print '------enter into init_edge_measure----'
        while 1:
            line = fp.readline()
            print line
            if not line:
                break
            line = line.split(',')
            param = []
            if len(line) == length:
                for k in range(length):
                    param.append(int(line[k]))
            self.set_edge_measure(*param)

    # 记录每个结点的相邻结点  like as follows:
    #  rec = {1:[2,3],  2:[3,4]} and so on
    def init_node_adjs(self):
        for row_num in range(self.node_num):
            for col_num in range(self.node_num):
                if self.connection_status[row_num][col_num]:
                    self.node_adjs[row_num].append(col_num)
        print self.node_adjs

    def set_measure(self, fp, length):
        self.init_edge_measure(fp, length)
        self.init_node_adjs()


class GlobalInfo:
    #Graph
    graph = None
    # Q1 / delat_delay
    Q1 = 100.0
    # Q2 / delat_cost
    Q2 = 400.0
    # 启发因子
    alpha = 1
    # 期望因子
    beta = 1
    # 信息素挥因子
    rho = 0.1
    # 如果随机数小于或等于r则选择max{pher(r,s)}
    # 否则按照概率公式进行转移
    r = 0.2
    # 当delay>delay_w时，将其乘以一个系数
    k = 0.5
    # pheromone[][] 全局信息素初始值
    C = 1
    node_num = 0
    edge_num = 0
    src_node = 0
    dst_node = 0
    ant_num = 0
    delay_w = 0
    Pc = 0
    Pm = 0
    pop_scale = 0
    pheromone = None
    delta_pheromone = None

    @staticmethod
    def init_param(node_num, edge_num, src_node, dst_node,
                   ant_num, delay_w, Pc, Pm, pop_scale):

        GlobalInfo.node_num = node_num
        GlobalInfo.edge_num = edge_num
        GlobalInfo.src_node = src_node
        GlobalInfo.dst_node = dst_node
        GlobalInfo.ant_num  = ant_num
        GlobalInfo.delay_w  = delay_w
        GlobalInfo.Pc       = Pc
        GlobalInfo.Pm       = Pm
        GlobalInfo.pop_scale= pop_scale
        # 全局信息素矩阵初始化
        GlobalInfo.pheromone = [[GlobalInfo.C for col in range(GlobalInfo.node_num)] for row in range(GlobalInfo.node_num)]
        # 全局信息素变化矩阵
        GlobalInfo.delta_pheromone = [[0 for col in range(GlobalInfo.node_num)] for row in range(GlobalInfo.node_num)]

    @staticmethod
    def read_info_from_file():
        # 从文件中读取信息并初始化GlobalInfo类和Graph类
        fp = open("test03.txt", 'r')
        line = fp.readline().split()
        node_num = int(line[0])
        edge_num = int(line[1])
        print 'node_num and edge_num: ', line
        fp.readline()
        line = fp.readline().split()
        src_node = int(line[0])
        dst_node = int(line[1])
        ant_num  = int(line[2])
        delay_w  = int(line[3])
        Pc       = float(line[4])
        Pm       = float(line[5])
        pop_scale= int(line[6])
        print 'src=', src_node
        print 'dst=', dst_node
        print 'ant_num=', ant_num
        print 'delay_w=', delay_w
        print 'Pc = ', Pc
        print 'Pm = ', Pm
        print 'pop_scale = ', pop_scale
        information = [node_num, edge_num, src_node, dst_node,
                       ant_num, delay_w, Pc, Pm, pop_scale]
        GlobalInfo.graph = Graph(node_num, edge_num)
        # 每条边相关的度量
        param_length     = 5
        GlobalInfo.graph.set_measure(fp,param_length)
        return information


class Chromosome:
    def __init__(self,):
        # 解
        self.solution = [GlobalInfo.src_node]
        # 适应度值
        self.fitness = 0

    def set_fitnes(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_solution(self, solution):
        self.solution = solution

    def get_solution(self):
        return self.solution

    def calculate_fitness_old(self, r=3, k1=100.0):
        i = 0
        path_length = len(self.solution)
        # 解对应的总时延
        sum_delay   = 0
        # 解对应的总花费
        sum_cost    = 0
        # Graph中delay Matrix
        tmp_delay   = GlobalInfo.graph.get_delay()
        # Graph中delay Matrix
        tmp_cost    = GlobalInfo.graph.get_cost()
        while i < path_length-1:
            sum_delay += tmp_delay[self.solution[i]][self.solution[i+1]]
            sum_cost  += tmp_cost[self.solution[i]][self.solution[i+1]]
            i += 1
        delta_sum_delay = GlobalInfo.delay_w - sum_delay
        print 'delta_sum_delay=',delta_sum_delay
        if delta_sum_delay >= 0:
            self.fitness = 1.0 * k1 / sum_cost
        else:
            self.fitness = 1.0 * k1 / (r * sum_cost)

    def calculate_fitness(self):
        pass

# 种群:
class Population:
    def __init__(self):
        # 种群规模
        self.pop_size = GlobalInfo.pop_scale
        # 染色体组
        self.chromosomes = [Chromosome() for num in range(self.pop_size)]
        # 最佳适应度
        self.best_fitness = 0
        # 最佳解
        self.best_solution = []
        # 平均适应度
        self.avg_fitness = 0
        # 交配池
        self.mate_pool = []

    # 随机生成一个解
    @staticmethod
    def random_solution():
        # 用于回溯
        stack = []
        # 解
        solution = [GlobalInfo.src_node]
        # 已访问结点
        visited = [GlobalInfo.src_node]
        current_node = GlobalInfo.src_node
        unvisited_adj_nodes = []
        adjs = GlobalInfo.graph.get_node_adjs()
        while True:
            # 相邻而且未访问
            print 'current_node=========',current_node
            for node in adjs[current_node]:
                if node not in visited:
                    print 'node=', node
                    unvisited_adj_nodes.append(node)
            if len(unvisited_adj_nodes) != 0:
                print 'unvisited_adj_nodes=', unvisited_adj_nodes
                next_node = choice(unvisited_adj_nodes)
                unvisited_adj_nodes = []
                print 'next_node = ', next_node
                solution.append(next_node)
                visited.append(next_node)
                stack.append(current_node)
                current_node = next_node
                if current_node == GlobalInfo.dst_node:
                    return  solution
            else:
                solution.pop()
                current_node = stack.pop()

    def init(self):
        for chromosome in self.chromosomes:
            chromosome.set_solution(Population.random_solution())

    def calculate_fitness(self):
        pass

    def choose(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass


if __name__ == "__main__":
    # 从文件中读取信息
    info = GlobalInfo.read_info_from_file()
    # 根据读取的信息初始化Graph类和GlobalInfo类
    GlobalInfo.init_param(*info)
    random_solution = Population.random_solution()
    print 'random_solution=', random_solution