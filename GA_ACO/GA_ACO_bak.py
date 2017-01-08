# coding=utf-8
# Author:   Lixinming
# 融合了遗传算法和蚁群算法的混合算法
# 关键在于如何融合：先遗传，在蚁群，蚁群算法的信息素初始值如何由遗传算法获得
from random import choice
from random import random
from random import uniform
from random import shuffle
from operator import attrgetter
from math import log10
import matplotlib.pyplot as pl

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

    def get_connection_status(self, row, col):
        return self.connection_status[row][col]

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
    def __init__(self):
        pass

    @staticmethod
    def get_delay(solution=[]):
        solution_len = len(solution)
        if solution_len == 0:
            return 0
        delay = 0
        delay_matrix  =GlobalInfo.graph.get_delay()
        for i in range(solution_len-1):
            delay += delay_matrix[solution[i]][solution[i+1]]
        return  delay

    # 用于标注何时进行算法的切换
    current_generation = 0
    generations = []
    best_fitnesses = []
    avg_fitnesses = []
    min_costs = []
    best_solution = []
    best_delay = 0
    # 用于标志是否是第一次进行种群适应度值计算，如果是第一次，则不
    count_calculate = 0
    # 表示如果GA算法中best_fitness连续STOP_TIME次停滞不变，则进入蚁群算法
    STOP_TIME = 3
    # GA算法最佳适应度值停滞次数
    stop_time = 0
    ACO_MAX_GENERATION = 100
    # GA算法最大迭代次数
    GA_MAX_GENERATION = 100
    #Graph
    graph = None
    # Q1 / delat_delay
    Q1 = 150.0
    # Q2 / delat_cost
    Q2 = 150.0
    # 启发因子
    alpha = 1
    # 期望因子
    beta = 1
    # 信息素挥因子
    rho = 0.2
    rho2 = rho
    # 如果随机数小于或等于r则选择max{pher(r,s)}
    # 否则按照概率公式进行转移
    r = 0.1
    # 当delay>delay_w时，将其乘以一个系数
    # GA算法中计算ftness时的系数 (对算法的收敛速度有影响)
    coef = 3
    # pheromone[][] 全局信息素初始值
    C = 0.31
    # pheromone[][]全局信息素的增量，这是由于遗传算法的解导致信息素的更新
    deltaC = 1
    k = 10
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
    # 惩罚系数
    punishment_coef = 0.7
    c1=0.8
    c2=1-c1
    result = 0
    # LOOP_TIME = 100
    LOOP_TIME = 1

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
        # fp = open("test03.txt", 'r')
        # fp = open("test03_new.txt", 'r')
        fp = open("test04.txt", 'r')
        line = fp.readline().split()
        node_num = int(line[0])
        edge_num = int(line[1])
        GlobalInfo.result = int(line[2])
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


"""
    遗传算法部分
"""
class Chromosome:
    def __init__(self,):
        # 解
        self.solution = [GlobalInfo.src_node]
        # 适应度值
        self.fitness = 0
        # 时延
        self.delays = 0
        # 花费
        self.cost = 0

    def calculate_cost(self):
        self.cost = 0
        cost_matirx = GlobalInfo.graph.get_cost()
        solution_len = len(self.solution)
        for i in range(solution_len - 1):
            self.cost += cost_matirx[self.solution[i]][self.solution[i+1]]

    def get_cost(self):
        if self.cost == 0:
            self.calculate_cost()
        return self.cost

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_delays(self,  delays):
        self.delays = delays

    def calculate_delays(self):
        solution_len = len(self.solution)
        delay_matrix = GlobalInfo.graph.get_delay()
        sum_delay = 0
        for i in range(solution_len-1):
            sum_delay += delay_matrix[self.solution[i]][self.solution[i+1]]
        self.delays = sum_delay

    def get_delays(self):
        if self.delays == 0:
            self.calculate_delays()
        return self.delays

    def set_solution(self, solution):
        self.solution = solution

    def get_solution(self):
        return self.solution

    def calculate_fitness_old(self, r=3, k1=1000.0):
        """
        功能： 计算个体的适应度值fitness,同时计算其总时延delays
        :param r:   惩罚系数
        :param k1:  协调系数
        :return:    None
        """
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
        total_cost = GlobalInfo.graph.get_total_cost()
        while i < path_length-1:
            sum_delay += tmp_delay[self.solution[i]][self.solution[i+1]]
            sum_cost  += tmp_cost[self.solution[i]][self.solution[i+1]]
            i += 1
        self.delays = sum_delay
        self.cost   = sum_cost
        delta_sum_delay = GlobalInfo.delay_w - sum_delay
        print 'delta_sum_delay=',delta_sum_delay

        if delta_sum_delay >= 0:
            self.fitness = (total_cost+0.0)/self.cost
        else:
            self.fitness = (total_cost+0.0)/(GlobalInfo.coef * sum_cost)

    def calculate_fitness(self):
        i = 0
        path_length = len(self.solution)
        sum_delay = 0
        sum_cost = 0
        while i < path_length - 1:
            tmp_delay = GlobalInfo.graph.get_delay()
            tmp_cost = GlobalInfo.graph.get_cost()
            sum_delay += tmp_delay[self.solution[i]][self.solution[i + 1]]
            sum_cost += tmp_cost[self.solution[i]][self.solution[i + 1]]
            i += 1
        print 'sum_delay = ', sum_delay
        delta_sum_delay = GlobalInfo.delay_w - sum_delay
        print '-------------delta_sum_delay=', delta_sum_delay
        if delta_sum_delay >= 0:
            self.fitness = (GlobalInfo.graph.total_cost - sum_cost) * \
                           log10(GlobalInfo.graph.total_delay + delta_sum_delay)
        else:
            self.fitness = pow(GlobalInfo.graph.total_cost - sum_cost, GlobalInfo.punishment_coef) * \
                           log10(GlobalInfo.graph.total_delay + delta_sum_delay)

    def solve_loop(self):
        exist_loop = 0
        contribute_loop_node = []
        # 注意此处必须使用node_num，而不是len(solution)
        # 比如solution=[0,1,1,4],则len(solution)=4,bucket[4]下标越界
        bucket = [0] * GlobalInfo.node_num
        print 'bucket=', bucket
        # 首先判断是否有环(重复结点)
        for node in self.solution:
            bucket[node] += 1
            if bucket[node] > 1:
                contribute_loop_node.append(node)
                exist_loop += 1
        # 如果没环（重复结点）
        if exist_loop == 0:
            print 'there is no loop'
            print self.solution
            return self
        # 否则有环
        solution = self.solution
        max_delta_index = 0
        # 如果有环,则去除(最大冗余环)
        for node in contribute_loop_node:
            # 先求重复结点的第一次出现的位置
            index1 = solution.index(node)
            # 再求重复结点的第二次出现的位置
            solution_len = len(solution)
            solution.reverse()
            index2 = solution_len - (solution.index(node) + 1)
            # 由于搞忘记还原了，导致出错了还不知所以然，冒失鬼,还是我测试屌炸天
            solution.reverse()
            delta_index = index2 - index1
            if max_delta_index < delta_index:
                max_delta_index = delta_index
                solution_pruned = solution[0:index1] + solution[index2:]
        self.solution = solution_pruned
        print 'after eliminate loop '
        print self.solution
        return self


    # 根据graph和变异结点，返回新的解solution
    def find_path(self, mutate_node):
        print '---------------enter into find_path-------'
        print 'mutate_node=', mutate_node
        node_adjs = GlobalInfo.graph.get_node_adjs()
        solution_len = len(self.solution)
        # 变异结点的索引（下标）
        mutate_node_index = self.solution.index(mutate_node)
        # 已访问结点
        visited = []

        # 可(未)访问结点
        tabu = []
        # 变异结点可以变异成的结点,此时至少有一个
        candidate = []
        for node in self.solution:
            visited.append(node)
        for node in range(GlobalInfo.graph.node_num):
            if node not in visited:
                tabu.append(node)
        for node in tabu:
            if GlobalInfo.graph.get_connection_status(node, self.solution[mutate_node_index - 1]):
                candidate.append(node)
        print 'candidate=', candidate
        # 变异结点变异后变成的结点
        if len(candidate) == 0:
            return self.solution
        current_node = choice(candidate)
        found = False
        tmp_solution = self.solution[0:mutate_node_index]
        while not found:
            tmp_solution.append(current_node)
            visited.append(current_node)
            for node in self.solution[-1:mutate_node_index:-1]:
                if GlobalInfo.graph.get_connection_status(node, current_node):
                    found = True
                    node_index = self.solution.index(node)
                    tmp_solution += self.solution[node_index:]
                    # tmp_solution.append(self.solution[solution_len - 1])
                    break
            if found:
                print '----------------------find wow wow!!!'
                return tmp_solution
            else:
                # 更新next_node,与next_node相邻而且没有访问过的结点
                current_node_adjs = []
                for node in node_adjs[current_node]:
                    if node not in visited:
                        current_node_adjs.append(node)
                if len(current_node_adjs) == 0:
                    return self.solution
                # 当前结点的相邻而未访问过的结点如果存在
                else:
                    current_node = choice(current_node_adjs)
        # default return value
        return self.solution

    def mutate(self):
        print '----------------------------enter into Chromosome mutate-----------'
        random_p = random()
        if random_p > GlobalInfo.Pm:
            return
        # 可能会变异的结点
        node_may_be_mutated = []
        solution_len = len(self.solution)
        # 随机选择一个去除首尾以后的片段中的一个结点，作为变异点
        print 'self.solution=',self.solution
        # 找到路径上除去首尾之外的结点
        for node in self.solution[1:solution_len-1]:
            node_may_be_mutated.append(node)
        # 随机选择一个作为变异结点
        print 'node_may_be_mutated=',node_may_be_mutated
        if len(node_may_be_mutated) == 0:
            return
        mutate_node = choice(node_may_be_mutated)
        self.solution = self.find_path(mutate_node)
        # self.solution = self.find_path_new(graph, mutate_node)
        self.solve_loop()


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
        # min_cost in accordance with best_fitness and best_solution
        self.min_cost = 0

    @staticmethod
    def random_solution_old():
        print 'enter into random_chromosome'
        graph = GlobalInfo.graph
        src_node = GlobalInfo.src_node
        des_node = GlobalInfo.dst_node
        visited = [src_node]
        # tabu 在此处表示可以访问的结点
        tabu = [n for n in range(graph.node_num)]
        tabu.remove(src_node)
        tabu_bak = tabu
        # 当前结点（位置）
        current_node = src_node
        # 下一个随机的相邻结点(位置)
        next_node = -1
        flag = False
        print 'aaaaaaaaa'
        print 'src_node', src_node
        print 'des_node', des_node
        # 定义栈用于回溯
        stack = []
        while len(tabu) > 0:
            print 'bbbbbbbb'
            # 当前结点的未访问的相邻结点列表
            adj_node_unvisited_list = []
            # 找到与current_node相邻结点中随机一个结点
            for node in range(graph.node_num):
                # 如果相邻结点还没有访问过，就加入带访问列表
                print '--------', node
                if graph.connection_status[current_node][node]:
                    print 'node=', node
                    print 'tabu=', tabu
                    if node in tabu:
                        print '-------append-----'
                        adj_node_unvisited_list.append(node)
                        print adj_node_unvisited_list
                    print '-------------not empty-------'
                else:
                    print '------continue----'
                    continue
            length = len(adj_node_unvisited_list)
            # 有相邻结点
            if length > 0:
                # next_node = adj_node_unvisited_list[randint(0, length-1)]
                next_node = choice(adj_node_unvisited_list)
                print 'next_node', next_node

                tabu.remove(next_node)
                visited.append(next_node)
                stack.append(current_node)
                current_node = next_node
                print 'find a adj node, current_node=', current_node
                if current_node == des_node:
                    flag = True
                    break
            # 没有相邻结点
            else:
                print 'cccccccc'
                print 'current_node=', current_node
                print 'visited=', visited
                visited.remove(current_node)
                current_node = stack.pop()
                # 重来一次
                # visited = [src_node]
                # tabu = tabu_bak
            next_node = -1
        if flag:
            print '------------find a valid path------'
            print visited
            return visited
        return None

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
        """
            功能：初始化种群中每个个体的解(solution)
        """
        print '---------------enter into init function of Population------'
        for chromosome in self.chromosomes:
            chromosome.set_solution(Population.random_solution())
            # 打印信息
        for chromosome in self.chromosomes:
            print chromosome.get_solution()
        # 先计算每个个体的适应度值，然后按照fitness进行从大到小排序
        # 计算最佳适应度值和平均适应度值


    def calculate_fitness(self):
        """
            功能：   计算适应度每个个体的适应度值,并得到最佳适应度值和对应解以及平均适应度值
            参数：   无
            返回值： 无
        """
        # 计算平均适应度值
        sum_fitness = 0
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness()
            print 'fitness as follows:',chromosome.get_fitness()
            sum_fitness += chromosome.get_fitness()
        self.avg_fitness = sum_fitness / self.pop_size
        # 按照适应度值进行从大大小排序并得到最佳适应度值以及对应的解
        # 刚开始没有将sorted的返回值赋值给self.chromosomes导致平均值比最大值还大，去死吧
        self.chromosomes = sorted(self.chromosomes,key=attrgetter('fitness'),reverse=True)
        best_chromosome    = self.chromosomes[0]
        best_fitness = best_chromosome.get_fitness()
        print 'best=',best_chromosome.get_fitness()
        #  如果下一代种群中最佳适应度值比当前的大，则更新最佳适应个体信息
        if best_fitness > self.best_fitness:
            self.best_fitness  = best_chromosome.get_fitness()
            self.best_solution = best_chromosome.get_solution()
            self.min_cost      = best_chromosome.get_cost()
            # 将GlobalInfo.stop_time清0
            GlobalInfo.stop_time = 0

        else:
            # 如果不超过当前最佳适应度值，则做标记
            GlobalInfo.stop_time += 1
        # if GlobalInfo.count_calculate > 0:
        # 当前全局最佳适应度值对应的解以及花费
        GlobalInfo.best_fitnesses.append(self.best_fitness)
        GlobalInfo.avg_fitnesses.append(self.avg_fitness)
        GlobalInfo.min_costs.append(self.min_cost)
        GlobalInfo.best_solution = self.best_solution
        GlobalInfo.best_delay = best_chromosome.get_delays()
        # Population.global_
        # GlobalInfo.count_calculate = 1
        # 打印信息
        print 'best_solution = ', self.best_solution
        print 'best_fitness = ',  self.best_fitness
        print 'avg_fitness = ',   self.avg_fitness
        print 'delays_of_best_solution = ', self.chromosomes[0].get_delays()
        print 'min_cost =', self.min_cost

    def choose(self):
        """
            功能：利用赌轮法选择pop_size个个体进入交配池,
                 同时交配池中的个体按照适应度值从大到小进行排序
        """
        print '--------------------------enter into choose function----------------------------'
        print 'self.best_fitness=', self.best_fitness
        print 'self.best_solution=', self.best_solution
        print 'self.pop_size=', self.pop_size
        # 将最佳适应度值对应的个体保留到交配池并进入下一代
        best_chromosome = Chromosome()
        best_chromosome.set_solution(self.best_solution)
        best_chromosome.set_fitness(self.best_fitness)
        self.mate_pool = [best_chromosome]
        for num in range(self.pop_size - 1):
            # 随机生成一个0到1之间的随机浮点数
            tmp_random = random()
            print 'tmp_random===', tmp_random
            sum_fitness = 0
            total_fitness = 0
            for int_i in range(self.pop_size):
                print 'self.pop_size = ',self.pop_size
                print 'int_i =', int_i
                total_fitness += self.chromosomes[int_i].get_fitness()
            for int_i in range(self.pop_size):
                sum_fitness += self.chromosomes[int_i].get_fitness() * 1.0 / total_fitness
                print 'sum_fitness=', sum_fitness
                if tmp_random <= sum_fitness:
                    print 'tmp_random=', tmp_random, ';sum_fitness=', sum_fitness
                    self.mate_pool.append(self.chromosomes[int_i])
                    break
        print "in choose length of mate_pool is:", len(self.mate_pool)
        # 对交配池中个体按照适应度值进行从大到小排序
        self.mate_pool.sort(key=attrgetter('fitness'), reverse=True)

        # 打印信息
        print 'len_mate_pool=', len(self.mate_pool)
        print 'mate_pool='
        for chrom in self.mate_pool:
            print chrom.get_solution()

    def crossover(self):
        """
        功能： 对交配池中的个体两两进行交叉操作,并放回交配池
        :return: None
        """
        print '---------enter into crossvoer-------'
        mate_pool_bak = []
        for i in range(self.pop_size / 2):
            pair = Population.static_crossover(self.mate_pool[i*2],self.mate_pool[i*2+1])

            #一个缩进高忘了，结果出错了，还调试了半条，真倒霉 WAF
            # 看来一定要细心呀，
            mate_pool_bak += pair
        self.mate_pool = mate_pool_bak
        print 'in crossover length of mate_pool is ', len(self.mate_pool)


    @staticmethod
    def static_crossover(chromosome1, chromosome2):
        """
        功能：输入两个个体进行交叉，输出两个个体
        :param chromosome1:个体1
        :param chromosome2:个体2
        :return:经过交叉操作之后的两个个体
        """
        print '---------enter into static_crossover---'
        chromosomes = [chromosome1, chromosome2]
        for chromosome in chromosomes:
            print chromosome.get_solution()
        if chromosome1.get_solution() == chromosome2.get_solution():
            return chromosomes
        rand_value = random()
        # 依照概率进行交叉
        if rand_value > GlobalInfo.Pc:
            return chromosomes
        res = []
        result = []
        same_points = []
        # 除去首尾，找到相同结点集，随机选择一个相同结点，相同结点之后进行交换
        for chromosome in chromosomes:
            solution = chromosome.get_solution()
            print 'before pop'
            print solution
            tmp_chromosome = Chromosome()
            # 去除首尾结点
            tmp_chromosome.set_solution(solution[1:len(solution)-1])
            res.append(tmp_chromosome)
        solution0 = res[0].get_solution()
        solution1 = res[1].get_solution()
        for node in solution0:
            if node in solution1:
                same_points.append(node)
        if len(same_points) == 0:
            # 没有相同结点,原样返回
            return chromosomes
        # 否则有相同结点，随机选择一个
        same_point = choice(same_points)
        print 'same_point=', same_point
        index0 = solution0.index(same_point)
        index1 = solution1.index(same_point)
        solution0_before_part = solution0[0:index0]
        solution0_after_part  = solution0[index0:]+[GlobalInfo.dst_node]

        solution1_before_part = solution1[0:index1]
        solution1_after_part  = solution1[index1:]+[GlobalInfo.dst_node]

        cost_matrix = GlobalInfo.graph.get_cost()
        delay_matrix = GlobalInfo.graph.get_delay()
        sum_cost0 = 0
        sum_cost1 = 0
        sum_delay0 = 0
        sum_delay1 = 0
        solution0_after_part_len = len(solution0_after_part)
        solution1_after_part_len = len(solution1_after_part)

        for i in range(solution0_after_part_len - 1):
            sum_cost0 += cost_matrix[solution0_after_part[i]][solution0_after_part[i+1]]
            sum_delay0 += delay_matrix[solution0_after_part[i]][solution0_after_part[i+1]]
        delta_sum_delay0 = GlobalInfo.delay_w - sum_delay0
        if delta_sum_delay0>=0:
            fitness0 = log10(GlobalInfo.graph.total_delay+delta_sum_delay0)*\
                       (GlobalInfo.graph.total_cost-sum_cost0)
        else:
            fitness0 = log10(GlobalInfo.graph.total_delay+delta_sum_delay0)* \
                       pow(GlobalInfo.graph.total_cost-sum_cost0,GlobalInfo.punishment_coef)

        for i in range(solution1_after_part_len - 1):
            sum_cost1 += cost_matrix[solution1_after_part[i]][solution1_after_part[i+1]]
            sum_delay1 += delay_matrix[solution1_after_part[i]][solution1_after_part[i+1]]
        delta_sum_delay1 = GlobalInfo.delay_w - sum_delay1
        if delta_sum_delay1>=0:
            fitness1 = log10(GlobalInfo.graph.total_delay+delta_sum_delay1)* \
                       (GlobalInfo.graph.total_cost-sum_cost1)
        else:
            fitness1 = log10(GlobalInfo.graph.total_delay+delta_sum_delay1)* \
                       pow(GlobalInfo.graph.total_cost-sum_cost1,GlobalInfo.punishment_coef)

        if fitness0>fitness1:
            # 谁的适应度值大，后半段就选择谁的
            solution1_after_part = solution0_after_part
        else:
            solution0_after_part = solution1_after_part

        chromosome0 = Chromosome()
        chromosome0.set_solution([GlobalInfo.src_node]+solution0_before_part+solution0_after_part)
        result.append(chromosome0)
        chromosome1 = Chromosome()
        chromosome1.set_solution([GlobalInfo.src_node]+solution1_before_part+solution1_after_part)
        result.append(chromosome1)

        for chromosome in result:
            print chromosome.get_solution()
        print '-----before solve loop'
        for chromosome in result:
            chromosome.solve_loop()
        return result

    @staticmethod
    def static_crossover_old(chromosome1, chromosome2):
        """
        功能：输入两个个体进行交叉，输出两个个体
        :param chromosome1:个体1
        :param chromosome2:个体2
        :return:经过交叉操作之后的两个个体
        """
        print '---------enter into static_crossover---'
        chromosomes = [chromosome1, chromosome2]
        for chromosome in chromosomes:
            print chromosome.get_solution()
        if chromosome1.get_solution() == chromosome2.get_solution():
            return chromosomes
        rand_value = random()
        # 依照概率进行交叉
        if rand_value > GlobalInfo.Pc:
            return chromosomes
        res = []
        result = []
        same_points = []
        # 除去首尾，找到相同结点集，随机选择一个相同结点，相同结点之后进行交换
        for chromosome in chromosomes:
            solution = chromosome.get_solution()
            print 'before pop'
            print solution
            tmp_chromosome = Chromosome()
            tmp_chromosome.set_solution(solution[1:len(solution) - 1])
            res.append(tmp_chromosome)
        solution0 = res[0].get_solution()
        solution1 = res[1].get_solution()
        for node in solution0:
            if node in solution1:
                same_points.append(node)
        if len(same_points) == 0:
            # 没有相同结点,原样返回
            return chromosomes
        # 否则有相同结点，随机选择一个
        same_point = choice(same_points)
        print 'same_point=', same_point
        index0 = solution0.index(same_point)
        index1 = solution1.index(same_point)
        solution0_before_part = solution0[0:index0]
        solution0_after_part = solution0[index0:]

        solution1_before_part = solution1[0:index1]
        solution1_after_part = solution1[index1:]
        chromosome0 = Chromosome()
        chromosome0.set_solution(
            [GlobalInfo.src_node] + solution0_before_part + solution1_after_part + [GlobalInfo.dst_node])
        result.append(chromosome0)
        chromosome1 = Chromosome()
        chromosome1.set_solution(
            [GlobalInfo.src_node] + solution1_before_part + solution0_after_part + [GlobalInfo.dst_node])
        result.append(chromosome1)

        for chromosome in result:
            print chromosome.get_solution()
        print '-----before solve loop'
        for chromosome in result:
            chromosome.solve_loop()
        return result

    def mutate(self):
        # 对交配池中的个体按照概率进行变异操作
        print 'length of self.mate_pool is ',len(self.mate_pool)
        for chromosome in self.mate_pool:
            chromosome.mutate()

    def update(self):
        print 'length of self.chromosomes is ',len(self.chromosomes)
        self.chromosomes = self.mate_pool
        print 'length of self.chromosomes is ',len(self.chromosomes)
        self.calculate_fitness()

    def solve(self):
        generation = 0
        MAX_GENERATION = GlobalInfo.GA_MAX_GENERATION
        while generation < MAX_GENERATION:
            # Population.generations.append(generation)
            population.choose()
            population.crossover()
            population.mutate()
            population.update()
            if GlobalInfo.stop_time == GlobalInfo.STOP_TIME:
                GlobalInfo.current_generation = generation
                break
            generation += 1


"""
    遗传算法的解转换为信息素的更新
"""

class Transaction:

    def __init__(self):
        pass

    @staticmethod
    def convert_solutions_to_delta_of_pheromone(population=Population()):
        # 对于每个个体：增加对应信息素矩阵的值
        for chromosome in population.chromosomes:
            solution = chromosome.get_solution()
            solution_len = len(solution)
            for i in range(solution_len - 1):
                GlobalInfo.pheromone[solution[i]][solution[i+1]] += GlobalInfo.deltaC*GlobalInfo.k
                GlobalInfo.pheromone[solution[i+1]][solution[i]] += GlobalInfo.deltaC*GlobalInfo.k


"""
    蚁群算法部分
"""
class Ant:

    def __init__(self):
        self.current_city = GlobalInfo.src_node
        self.next_city    = -1
        # 允许访问的结点(城市）
        self.allowed = [node for node in range(GlobalInfo.node_num)]
        self.allowed.remove(self.current_city)
        print 'init self.allowed=',self.allowed
        # 解
        self.solution = [self.current_city]
        # 栈 用于找不到路径时回溯
        self.Stack = []
        # 适应度值
        self.fitness  = 0
        # 花费
        self.cost     = 0
        # 时延和
        self.delays   = 0

    def get_current_city(self):
        return self.current_city

    def set_current_city(self, current_city):
        self.current_city = current_city

    def set_solution(self, solution):
        self.solution = solution

    def get_solution(self):
        return self.solution

    def set_fitness(self, fitness_value):
        self.fitness = fitness_value

    def calculate_fitness(self):
        # self.fitness = log10(GlobalInfo.graph.get_total_delay() + GlobalInfo.delay_w - self.delays) \
        #                * (GlobalInfo.graph.get_total_cost() - self.cost)
        i = 0
        path_length = len(self.solution)
        sum_delay = 0
        sum_cost = 0
        while i < path_length - 1:
            tmp_delay = GlobalInfo.graph.get_delay()
            tmp_cost = GlobalInfo.graph.get_cost()
            sum_delay += tmp_delay[self.solution[i]][self.solution[i + 1]]
            sum_cost += tmp_cost[self.solution[i]][self.solution[i + 1]]
            i += 1
        print 'sum_delay = ', sum_delay
        delta_sum_delay = GlobalInfo.delay_w - sum_delay

        if delta_sum_delay >= 0:
            self.fitness = (GlobalInfo.graph.total_cost - sum_cost) * \
                           log10(GlobalInfo.graph.total_delay + delta_sum_delay)
        else:
            self.fitness = pow(GlobalInfo.graph.total_cost - sum_cost, GlobalInfo.punishment_coef) * \
                           log10(GlobalInfo.graph.total_delay + delta_sum_delay)


    def calculate_fitness_old(self):
        total_cost  = GlobalInfo.graph.get_total_cost()
        cost = self.get_cost()
        print 'cost=',cost
        print 'type(cost)=', type(cost)
        if self.get_delays() <= GlobalInfo.delay_w:
            self.fitness = (total_cost + 0.0)/cost
        else:
            self.fitness = (total_cost + 0.0)/(GlobalInfo.coef * cost)

    def get_fitness(self):
        return self.fitness

    def calculate_cost(self):
        """
            功能： 根据每只蚂蚁的解，计算每只蚂蚁对应的花费
        :return:
        """
        print '-----enter into calculate_cost()_--'
        cost_matrix = GlobalInfo.graph.get_cost()
        print 'cost_matrix=',cost_matrix
        solution_len = len(self.solution)
        print 'solution_len=', solution_len
        for i in range(solution_len - 1):
            self.cost += cost_matrix[self.solution[i]][self.solution[i+1]]
        print 'self.cost=', self.cost

    # 获取花费
    def get_cost(self):
        print '-----enter into get_cost()----'
        if self.cost == 0:
            print 'self.cost==0'
            self.calculate_cost()
        return self.cost

    def calculate_delays(self):
        solution_len = len(self.solution)
        delay_matrix = GlobalInfo.graph.get_delay()
        for i in range(solution_len - 1):
            self.delays += delay_matrix[self.solution[i]][self.solution[i+1]]

    def get_delays(self):
        if self.delays == 0:
            self.calculate_delays()
        return self.delays

    def choose_next_city(self):
        """
            功能：选择下一个城市
            :return:
        """
        print '--------enter into choose_next_city()---'
        # 计算路径迁移概率
        transe_p = {}
        max_value = 0
        max_node = -1
        sum_value = 0
        print 'self.allowed=',self.allowed
        for city_index in self.allowed:
            # 在允许访问的结点中只需要考虑相邻结点（城市)
            if not GlobalInfo.graph.get_connection_status(self.current_city, city_index):
                continue
            # 计算分子
            fz= pow(GlobalInfo.pheromone[self.current_city][city_index], GlobalInfo.alpha) * \
                 (pow(1.0 / GlobalInfo.graph.get_cost()[self.current_city][city_index], GlobalInfo.beta))
            sum_value += fz
            if max_value < fz:
                max_value = fz
                max_node = city_index
            transe_p[city_index] = fz
        # 获取最大概率对应的结点(城市)编号
        rand_value = random()
        # 如果随机数不大于r,则选择概率最大的；否则以概率选择(有的论文排了序，有的没有排序)
        if rand_value <= GlobalInfo.r:
            print 'rand_value <= GlobalInfo.r:'
            self.next_city = max_node
            return
        rand_prob = uniform(0, sum_value)
        print 'sum_value=', sum_value
        print 'rand_prob=', rand_prob
        sum_prob = 0
        for k, v in transe_p.items():
            sum_prob += v
            print 'sum_prob=', sum_prob
            if sum_prob >= rand_prob:
                self.next_city = k
                print 'self.next_city=', self.next_city
                break

    def move_to_next_city(self):
        """
        功能:转移到下一个城市
        :return:
        """
        print '-----------enter into move_to_next_city()------'
        # 没找到下一个结点, 需要回溯
        if self.next_city == -1:
            self.solution.pop()
            self.current_city = self.Stack.pop()
        # 找到下一个结点
        else:
            self.Stack.append(self.current_city)
            self.current_city = self.next_city
            self.next_city = -1
            self.allowed.remove(self.current_city)
            self.solution.append(self.current_city)

    def find_path(self):
        """
        功能： 蚂蚁找路径
        :return:
        """
        while True:
            self.choose_next_city()
            self.move_to_next_city()
            if self.current_city == GlobalInfo.dst_node:
                break
            else:
                print 'current_city=', self.current_city

        # self.update_pheromone()
        print 'self.solution = ', self.solution

    def update_pheromone(self):
        """
        功能： 蚂蚁找到一条路径之后，就更新信息素
        :return:
        """
        print '---------enter into update_pheromone-----'
        path_length = len(self.solution)
        delay = GlobalInfo.graph.get_delay()
        cost = GlobalInfo.graph.get_cost()
        print 'delay=', delay
        print 'solution before update pheromone=', self.solution
        # 只更新时延约束满足的路径
        if delay <= GlobalInfo.delay_w:
            for i in range(path_length - 1):
                row_num = self.solution[i]
                col_num = self.solution[i + 1]
                if delay[row_num][col_num] == 0:
                     print '00000000000000000'
                     continue
                delta_tao = 1.0 / (GlobalInfo.c1*delay[row_num][col_num]+GlobalInfo.c2*cost[row_num][col_num])
                GlobalInfo.pheromone[row_num][col_num] *= (1 - GlobalInfo.rho)
                # GlobalInfo.pheromone[row_num][col_num] += GlobalInfo.rho * delta_tao * GlobalInfo.Q1
                GlobalInfo.pheromone[row_num][col_num] +=  delta_tao * GlobalInfo.Q1
                GlobalInfo.pheromone[col_num][row_num] = GlobalInfo.pheromone[row_num][col_num]



class AntSystem:
    def __init__(self):
        self.ant_num = GlobalInfo.ant_num
        self.ants = [Ant() for i in range(self.ant_num)]
        self.best_fitness = GlobalInfo.best_fitnesses[-1]
        self.avg_fitness  = GlobalInfo.avg_fitnesses[-1]
        self.best_solution= GlobalInfo.best_solution
        self.best_delay   = GlobalInfo.get_delay(self.best_solution)
        self.best_cost    = GlobalInfo.min_costs[-1]

    def find_path(self):
        for ant in self.ants:
            ant.find_path()
        print '----------ants=--------'
        for ant in self.ants:
            print ant.get_solution()

    def calculate_best_fitness_and_best_solution(self):
        sum_fitness = 0
        for ant in self.ants:
            ant.calculate_fitness()
            fitness = ant.get_fitness()
            if self.best_fitness < fitness:
                self.best_fitness = fitness
                self.best_solution = ant.solution
                self.best_cost = ant.get_cost()
                self.best_delay = ant.get_delays()
            sum_fitness += fitness
        self.avg_fitness = sum_fitness / GlobalInfo.ant_num

    def update_individual_pheromone(self):
        for ant in self.ants:
            ant.update_pheromone()

    def update_global_pheromone(self):
        path_length = len(self.best_solution)
        for i in range(path_length - 1):
            row_num = self.best_solution[i]
            col_num = self.best_solution[i + 1]
            cost_matrix = GlobalInfo.graph.get_cost()
            delta_tao = 1.0 / cost_matrix[row_num][col_num]

            GlobalInfo.pheromone[row_num][col_num] *= (1 - GlobalInfo.rho2)
            # GlobalInfo.pheromone[row_num][col_num] += GlobalInfo.rho2 * delta_tao * GlobalInfo.Q2
            GlobalInfo.pheromone[row_num][col_num] +=  delta_tao * GlobalInfo.Q2
            GlobalInfo.pheromone[col_num][row_num] = GlobalInfo.pheromone[row_num][col_num]

    def solve(self, best_fitnesses, avg_fitnesses, min_costs):
        generation = 0
        while generation < GlobalInfo.ACO_MAX_GENERATION:
            self.find_path()
            self.calculate_best_fitness_and_best_solution()
            print 'before update pheromone GlobalInfo.pheromone=', GlobalInfo.pheromone
            self.update_individual_pheromone()
            print 'after update pheromone GlobalInfo.pheromone=', GlobalInfo.pheromone
            self.update_global_pheromone()
            print '----------------------------------'
            print 'generation = ', generation
            print 'current best_fitness = ', self.best_fitness
            print 'current best_solution = ', self.best_solution
            cost_matrix = GlobalInfo.graph.get_cost()
            cost_sum = 0
            len_best_solutoin = len(self.best_solution)
            for i in range(len_best_solutoin-1):
                cost_sum += cost_matrix[self.best_solution[i]][self.best_solution[i+1]]
            print 'best_cost=',cost_sum
            print 'current best_cost = ', self.best_cost
            print 'current best_delay = ', self.best_delay
            best_fitnesses.append(self.best_fitness)
            avg_fitnesses.append(self.avg_fitness)
            min_costs.append(self.best_cost)
            GlobalInfo.best_delay = self.best_delay
            GlobalInfo.best_solution = self.best_solution
            self.ants = [Ant() for num in range(self.ant_num)]
            generation += 1
            if generation > 0:
                # 保证第一次不会出现最优解,更好地体现算法的性能
                GlobalInfo.count = 1


if __name__ == "__main__":
    # 从文件中读取信息
    info = GlobalInfo.read_info_from_file()
    # 根据读取的信息初始化Graph类和GlobalInfo类
    GlobalInfo.init_param(*info)
    population = Population()
    time = 0
    rate = 0
    avg_iteration_time = 0
    while time < GlobalInfo.LOOP_TIME:
        population.init()
        population.calculate_fitness()
        population.solve()

        # print '================='
        # print 'Population.generations=',GlobalInfo.generations
        # print 'Population.best_fitnesses=', GlobalInfo.best_fitnesses
        # print 'Population.avg_fitnesses=', GlobalInfo.avg_fitnesses
        # print 'Population.min_costs=', GlobalInfo.min_costs
        Transaction.convert_solutions_to_delta_of_pheromone()
        ant_system = AntSystem()
        # Population.generations += [i+GlobalInfo.STOP_TIME for i in range(GlobalInfo.ACO_MAX_GENERATION)]
        # delta_generation = Population.generations.pop()
        # Population.generations += [i+delta_generation+1 for i in range(GlobalInfo.ACO_MAX_GENERATION)]
        ant_system.solve(GlobalInfo.best_fitnesses, GlobalInfo.avg_fitnesses,
                         GlobalInfo.min_costs)
        generation_length = len(GlobalInfo.best_fitnesses)

        GlobalInfo.generations = [i for i in range(generation_length)]
        # value = (GlobalInfo.node_num, GlobalInfo.edge_num, GlobalInfo.pop_scale,GlobalInfo.Pc,
        #          GlobalInfo.Pm, GlobalInfo.ant_num,GlobalInfo.alpha,GlobalInfo.beta, GlobalInfo.rho, GlobalInfo.r, GlobalInfo.Q1,
        #          GlobalInfo.Q2, min(GlobalInfo.min_costs), GlobalInfo.best_delay,GlobalInfo.best_solution,GlobalInfo.STOP_TIME
        #          )
        # info = 'node_num=%d, edge_num=%d,pop_scale=%d,Pc=%.3f,Pm=%.3f, ant_num=%d,alpha=%d,beta=%d ' \
        #        'rho=%.2f, r=%.2f,Q1=%.2f, Q2=%.2f, min_cost=%d, ' \
        #        'best_delay=%d, best_solution=%s, stagnancy_time=%d' % value
        min_cost = GlobalInfo.min_costs[-1]
        if min_cost == GlobalInfo.result:
            rate += 1

        location = len(GlobalInfo.best_fitnesses) - 1
        indexes = [i for i in range(location + 1)]
        iter = 0
        for index in indexes[-1:0:-1]:
            if GlobalInfo.best_fitnesses[index] == GlobalInfo.best_fitnesses[index - 1]:
                iter += 1
            else:
                break
        iter_time = location - iter
        avg_iteration_time += iter_time
        time += 1
    # output ration and iter_time
    ration = rate * 100.0 / GlobalInfo.LOOP_TIME
    iter_time = avg_iteration_time * 1.0 / GlobalInfo.LOOP_TIME
    f = open("result","a+")
    # specific params from GA and ACO
    result = "some params "+str(ration)+"\t"+str(iter_time)+"\t"
    f.write(result)
    #pl.figure(info)
    #pl.subplot(211)
    ## pl.xlabel('generation')
    #pl.ylabel('fitness')
    #pl.plot(GlobalInfo.generations, GlobalInfo.best_fitnesses, 'r.-', label='best_fitness')
    #pl.plot(GlobalInfo.generations, GlobalInfo.avg_fitnesses, 'b.-', label='avg_fitness')
    ## pl.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    #pl.legend(loc='lower right')
    #pl.grid()
    #xAxis = GlobalInfo.current_generation
    #yAxis = GlobalInfo.best_fitnesses[GlobalInfo.current_generation]
    #if yAxis > 10:
    #    yAxis2 = yAxis - 5
    #else:
    #    yAxis2 = yAxis + 5
    #xAxis2 = xAxis + 20
    # pl.annotate('GA->ACO k=%s,switch_generation=%d' % (GlobalInfo.k,GlobalInfo.current_generation), xy=(xAxis, yAxis), xytext=(xAxis2, yAxis2), \
    #             arrowprops=dict(facecolor='green', shrink=0.1))
    #pl.subplot(212)
    #pl.xlabel('generation')
    #pl.ylabel('cost')
    #pl.plot(GlobalInfo.generations, GlobalInfo.min_costs, 'r.-', label='min_cost')
    #pl.grid()
    #pl.legend()
    #xAxis3 = xAxis
    #yAxis3 = GlobalInfo.min_costs[xAxis3]
    #xAxis4 = xAxis3
    #yAxis4 = yAxis3
    #pl.annotate('GA->ACO switch_generation=%d'%GlobalInfo.current_generation, xy=(xAxis3, yAxis3), xytext=(xAxis4, yAxis4), \
    #            arrowprops=dict(facecolor='green', shrink=0.5))
    #
    #pl.show()
    #
    ## ant = Ant()
    ## print 'aaaaaaaa'
    ## ant.find_path()
    ## ant.update_pheromone()