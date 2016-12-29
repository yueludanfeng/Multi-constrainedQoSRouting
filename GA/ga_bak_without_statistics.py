# coding=utf-8
# Author:   Li xinming
# coding=utf-8
from math import log10
from math import pow
from operator import attrgetter
from random import choice
from random import random
from random import sample
from time import clock

import matplotlib.pyplot as pl


# 输出重定向至指定的文件中，便于查看
# file_obj = open('out.txt', 'w+')
# stdout = file_obj


class Graph:
    def __init__(self, node_num=0, edge_num=0):
        self.node_num = node_num
        self.edge_num = edge_num
        self.connection_status = [[False for col in range(self.node_num)] for row in range(self.node_num)]
        self.bandwidth = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        self.delay = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        self.cost = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        self.node_adjs = {r:[] for r in range(self.node_num)}
        self.total_cost = 0
        self.total_delay= 0

    def get_node_adjs(self):
        return self.node_adjs

    def get_delay(self):
        return self.delay

    def get_cost(self):
        return self.cost

    def add_connection_status(self, rownum, colnum):
        self.connection_status[rownum][colnum] = True
        self.connection_status[colnum][rownum] = True

    def get_connection_status(self, rownum, colnum):
        return self.connection_status[rownum][colnum]

    def add_bandwidth(self, rownum, colnum, value = 0):
        self.bandwidth[rownum][colnum] = value
        self.bandwidth[colnum][rownum] = value

    def add_cost(self, rownum, colnum, value = 0):
        self.cost[rownum][colnum] = value
        self.cost[colnum][rownum] = value
        self.total_cost += value

    def add_delay(self, rownum, colnum, value = 0):
        self.delay[rownum][colnum] = value
        self.delay[colnum][rownum] = value
        self.total_delay += value

    def add_edge_measure(self, rownum, colnum, bandwidth, delay, cost):
        self.add_bandwidth(rownum, colnum, bandwidth)
        self.add_delay(rownum, colnum, delay)
        self.add_cost(rownum, colnum, cost)
        self.add_connection_status(rownum, colnum)

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

            self.add_edge_measure(*param)
            # self.add_edge_measure(param[0], param[1], param[2], param[3], param[4])
            # self.add_edge_measure(0, 1, 1, 2, 3)
            # self.add_edge_measure(0, 2, 4, 5, 7)
            # self.add_edge_measure(1, 3, 4, 5, 6)
            # self.add_edge_measure(1, 4, 3, 4, 5)
            # self.add_edge_measure(2, 4, 2, 3, 6)
            # self.add_edge_measure(3, 4, 7, 8, 9)
            # self.add_edge_measure(2, 3, 2, 4, 1)
            # self.init_node_adjs()

    # 记录每个结点的相邻结点  like as follows:     rec = {1:[2,3],  2:[3,4]} and so on
    def init_node_adjs(self):
        for row_num in range(self.node_num):
            for col_num in range(self.node_num):
                if self.connection_status[row_num][col_num]:
                    self.node_adjs[row_num].append(col_num)
        print self.node_adjs

class Global:
    coef = 0.5
    r = 2
    delay_w = 0
    Pm = 0
    k = 1000
class Chromosome:

    # 无参构造方法 也必须使用括号 eg: chromosome = Chromosome()
    def __init__(self):
        self.solution = []
        self.fitness = 0

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, value):
        self.fitness = value

    def set_solution(self, solution=None):
        if solution is None:
            solution = []
        self.solution = solution

    def get_solution(self):
        return self.solution

    def get_total_cost(self, graph):
        print '--------enter into get_total_cost() of Chromosome----'
        print 'self.solution = '
        print self.solution
        solution_len = len(self.solution)
        total_cost = 0
        print '=========cost_matrix================'
        print graph.get_cost()
        cost_matrix = graph.get_cost()
        # for node_num in self.solution[0:solution_len-1]:
        for k in range(solution_len-1):
            print 'cost[k][k+1]=',cost_matrix[self.solution[k]][self.solution[k+1]]
            total_cost += cost_matrix[self.solution[k]][self.solution[k+1]]
        print 'cost=',total_cost
        return total_cost

    def calculate_fitness(self, graph, delay_w):
        i = 0
        path_length = len(self.solution)
        sum_delay = 0
        sum_cost = 0
        tmp_delay = graph.get_delay()
        tmp_cost = graph.get_cost()
        while i < path_length - 1:
            sum_delay += tmp_delay[self.solution[i]][self.solution[i + 1]]
            sum_cost  += tmp_cost[self.solution[i]][self.solution[i + 1]]
            i += 1
        print 'delay_w = ', delay_w
        print 'sum_delay = ', sum_delay
        delta_sum_delay = delay_w - sum_delay
        print '-------------delta_sum_delay=',delta_sum_delay
        if delta_sum_delay >= 0:
            self.fitness = (graph.total_cost-sum_cost) *\
                           log10(graph.total_delay+delta_sum_delay)
        else:
            self.fitness = pow(graph.total_cost - sum_cost, Global.coef) * \
                            log10(graph.total_delay+delta_sum_delay)

        # if delta_sum_delay > 0:
        #     self.fitness = delta_sum_delay/sum_cost
        # else:
        #     self.fitness = sum_cost / 2
        # print '----delta_sum_delay>=0  ', (graph.total_cost-sum_cost) * log10(graph.total_delay+delta_sum_delay)
        # self.fitness = (graph.total_cost-sum_cost) * log10(graph.total_delay+delta_sum_delay)
        # if delta_sum_delay > 0:
        #     print '----delta_sum_delay>=0  ', (graph.total_cost-sum_cost) * log10(graph.total_delay+delta_sum_delay)
        #     self.fitness = (graph.total_cost-sum_cost) * log10(graph.total_delay+delta_sum_delay)
        # else:
        #      print '----delta_sum_delay<0  ', (graph.total_cost-sum_cost) / 2
        #      self.fitness = (graph.total_cost-sum_cost) / 2

        #if delta_sum_delay < 0:
        #if delta_sum_delay < 0:
        #    print 'np.math.exp(delta_sum_delay)=',np.math.exp(delta_sum_delay)
        #    self.fitness = np.math.exp(delta_sum_delay)
        #else:
        #    # print '----delta_sum_delay>=0  ',np.math.exp(C-sum_cost) * np.math.log10(10+delta_sum_delay)
        #    # self.fitness = np.math.exp(C-sum_cost) * np.math.log10(10+delta_sum_delay)
        #    print '----delta_sum_delay>=0  ',(C-sum_cost) * np.math.log10(10+delta_sum_delay)
        #    self.fitness = (C-sum_cost) * np.math.log10(10+delta_sum_delay)

    def calculate_fitness_old(self, graph, delay_w):
        i = 0
        path_length = len(self.solution)
        sum_delay = 0
        sum_cost = 0
        tmp_delay = graph.get_delay()
        tmp_cost = graph.get_cost()
        while i < path_length-1:
            sum_delay += tmp_delay[self.solution[i]][self.solution[i+1]]
            sum_cost += tmp_cost[self.solution[i]][self.solution[i+1]]
            i += 1
        delta_sum_delay = delay_w - sum_delay
        print 'delta_sum_delay=',delta_sum_delay
        if delta_sum_delay >= 0:
            self.fitness = 1.0 * Global.k / sum_cost
        else:
            self.fitness = 1.0 * Global.k / (Global.r *sum_cost)
        # if delta_sum_delay <= 0:
        #     delta_sum_delay = r
        # fz = k1 * delta_sum_delay
        # print fz/sum_cost
        # 之前fitness没有在此处赋值，导致每个个体的适应度值总是为0
        # self.fitness = fz/sum_cost

    def crossover_improv2(self, chromosome2, graph, pc):
        '''
        根据后半段的适应度绝决定用哪段
        :param chromosome2:
        :param graph:
        :param pc:
        :return:
        '''
        print '----------------------------enter into cross function of Chromosome-----------------'
        p = random()
        res = []
        chromosome1 = self
        if p > pc:
            res.append(chromosome1)
            res.append(chromosome2)
            return res
        # 记录公共点集合
        common_point_record = []
        # 标志是否存在公共点
        existence = 0
        solution1 = chromosome1.get_solution()
        solution2 = chromosome2.get_solution()
        len_solution1 = len(solution1)
        len_solution2 = len(solution2)
        for g in solution1[1:len_solution1 - 1]:
            if g in solution2[1:len_solution2 - 1]:
                existence += 1
                common_point_record.append(g)
        # 如果没有相同的结点，则不交叉，直接将两个染色体返回;
        # 如果有相同的结点，随机选择一个公共结点进行交换
        if existence != 0:
            random_common_point = choice(common_point_record)
            # 根据公共结点找到起分别所在染色体1和染色体2中的索引（位置）
            common_point_index_in_solution1 = solution1.index(random_common_point)
            common_point_index_in_solution2 = solution2.index(random_common_point)
            # 公共结点之前的片段
            solution1_before = solution1[0:common_point_index_in_solution1]
            solution2_before = solution2[0:common_point_index_in_solution2]
            # 公共结点之后的片段
            solution1_after = solution1[common_point_index_in_solution1:len_solution1]
            solution2_after = solution2[common_point_index_in_solution2:len_solution2]
            # 计算后半段的花费和，谁的小就用谁的代替原来的
            len_solution1_after = len(solution1_after)
            len_solution2_after = len(solution2_after)
            cost_matrix = graph.get_cost()
            delay_matrix = graph.get_delay()
            # 累计时延和花费
            sum_cost_of_solution1_after = 0
            sum_cost_of_solution2_after = 0
            sum_delay_of_solution1_after = 0
            sum_delay_of_solution2_after = 0
            for i in range(len_solution1_after - 1):
                sum_cost_of_solution1_after += cost_matrix[solution1_after[i]][solution1_after[i + 1]]
                sum_delay_of_solution1_after += delay_matrix[solution1_after[i]][solution1_after[i + 1]]
            for i in range(len_solution2_after - 1):
                sum_cost_of_solution2_after += cost_matrix[solution2_after[i]][solution2_after[i + 1]]
                sum_delay_of_solution2_after += delay_matrix[solution2_after[i]][solution2_after[i + 1]]
            delta_sum_delay1 = delay_w - sum_delay_of_solution1_after
            delta_sum_delay2 = delay_w - sum_delay_of_solution2_after
            if delta_sum_delay1 >= 0:
                fitness1 = (graph.total_cost - sum_cost_of_solution1_after) * \
                               log10(graph.total_delay + delta_sum_delay1)
            else:
                fitness1 = pow(graph.total_cost - sum_cost_of_solution1_after, Global.coef) * \
                               log10(graph.total_delay + delta_sum_delay1)
            if delta_sum_delay2 >= 0:
                fitness2 = (graph.total_cost - sum_cost_of_solution2_after) * \
                               log10(graph.total_delay + delta_sum_delay2)
            else:
                fitness2 = pow(graph.total_cost - sum_cost_of_solution2_after, Global.coef) * \
                               log10(graph.total_delay + delta_sum_delay2)
            if fitness1 > fitness2:
                solution2_after = solution1_after
            else:
                solution1_after = solution2_after

            solution1 = solution1_before + solution1_after
            solution2 = solution2_before + solution2_after
            chromosome1.set_solution(solution1)
            chromosome2.set_solution(solution2)
            list_chromosome = [chromosome1, chromosome2]
            for chromosome in list_chromosome:
                res.append(chromosome.solve_loop(graph))
        else:
            # 记录结果并返回
            res.append(chromosome1)
            res.append(chromosome2)
        return res

    def crossover_improv1(self, chromosome2, graph, pc):
        '''
        根据后半段的花费决定用那段
        :param chromosome2:
        :param graph:
        :param pc:
        :return:
        '''
        print '----------------------------enter into cross function of Chromosome-----------------'
        p = random()
        res = []
        chromosome1 = self
        if p > pc:
            res.append(chromosome1)
            res.append(chromosome2)
            return res
        # 记录公共点集合
        common_point_record = []
        # 标志是否存在公共点
        existence = 0
        solution1 = chromosome1.get_solution()
        solution2 = chromosome2.get_solution()
        len_solution1 = len(solution1)
        len_solution2 = len(solution2)
        for g in solution1[1:len_solution1 - 1]:
            if g in solution2[1:len_solution2 - 1]:
                existence += 1
                common_point_record.append(g)
        # 如果没有相同的结点，则不交叉，直接将两个染色体返回;
        # 如果有相同的结点，随机选择一个公共结点进行交换
        if existence != 0:
            random_common_point = choice(common_point_record)
            # 根据公共结点找到起分别所在染色体1和染色体2中的索引（位置）
            common_point_index_in_solution1 = solution1.index(random_common_point)
            common_point_index_in_solution2 = solution2.index(random_common_point)
            # 公共结点之前的片段
            solution1_before = solution1[0:common_point_index_in_solution1]
            solution2_before = solution2[0:common_point_index_in_solution2]
            # 公共结点之后的片段
            solution1_after = solution1[common_point_index_in_solution1:len_solution1]
            solution2_after = solution2[common_point_index_in_solution2:len_solution2]
            # 计算后半段的花费和，谁的小就用谁的代替原来的
            len_solution1_after = len(solution1_after)
            len_solution2_after = len(solution2_after)
            cost_matrix = graph.get_cost()
            sum_cost_of_solution1_after = 0
            sum_cost_of_solution2_after = 0
            for i in range(len_solution1_after-1):
                sum_cost_of_solution1_after += cost_matrix[solution1_after[i]][solution1_after[i+1]]
            for i in range(len_solution2_after-1):
                sum_cost_of_solution2_after += cost_matrix[solution2_after[i]][solution2_after[i+1]]
            if sum_cost_of_solution1_after > sum_cost_of_solution2_after:
                solution2_after = solution1_after
            else:
                solution1_after = solution2_after
            solution1 = solution1_before + solution1_after
            solution2 = solution2_before + solution2_after
            chromosome1.set_solution(solution1)
            chromosome2.set_solution(solution2)
            list_chromosome = [chromosome1, chromosome2]
            for chromosome in list_chromosome:
                res.append(chromosome.solve_loop(graph))
        else:
            # 记录结果并返回
            res.append(chromosome1)
            res.append(chromosome2)
        return res

    def crossover(self, chromosome2, graph, pc):
        print '----------------------------enter into cross function of Chromosome-----------------'
        p = random()
        res = []
        chromosome1 = self
        if p > pc:
            print 'p>self.pc'
            res.append(chromosome1)
            res.append(chromosome2)
            return res
        # 记录公共点集合
        common_point_record = []
        # 标志是否存在公共点
        existence = 0
        solution1 = chromosome1.get_solution()
        print 'solution1=', solution1
        solution2 = chromosome2.get_solution()
        print 'solution2=', solution2
        len_solution1 = len(solution1)
        len_solution2 = len(solution2)
        for g in solution1[1:len_solution1 - 1]:
            if g in solution2[1:len_solution2 - 1]:
                existence += 1
                common_point_record.append(g)
        # 如果没有相同的结点，则不交叉，直接将两个染色体返回;
        # 如果有相同的结点，随机选择一个公共结点进行交换
        if existence != 0:
            random_common_point = choice(common_point_record)
            print 'random_common_point=', random_common_point
            # 根据公共结点找到起分别所在染色体1和染色体2中的索引（位置）
            common_point_index_in_solution1 = solution1.index(random_common_point)
            print common_point_index_in_solution1
            common_point_index_in_solution2 = solution2.index(random_common_point)
            print common_point_index_in_solution2
            # 交换公共结点后面的片段
            # 公共结点之前的片段
            solution1_before = solution1[0:common_point_index_in_solution1]
            solution2_before = solution2[0:common_point_index_in_solution2]
            # 公共结点之后的片段
            solution1_after = solution1[common_point_index_in_solution1:len_solution1]
            solution2_after = solution2[common_point_index_in_solution2:len_solution2]
            # 交换并更新染色体
            print 'solution1_before=', solution1_before
            print 'solution1_after=', solution1_after
            print 'solution2_before=', solution2_before
            print 'solution2_after=', solution2_after
            solution1 = solution1_before + solution2_after
            solution2 = solution2_before + solution1_after
            print 'after crossover, solution1=', solution1
            print 'after crossover, solution2=', solution2
            chromosome1.set_solution(solution1)
            chromosome2.set_solution(solution2)
            list_chromosome = [chromosome1, chromosome2]
            for chromosome in list_chromosome:
                res.append(chromosome.solve_loop(graph))
        else:
            # 记录结果并返回
            res.append(chromosome1)
            res.append(chromosome2)
        return res

    def solve_loop(self, graph):
        exist_loop = 0
        contribute_loop_node = []
        # 注意此处必须使用node_num，而不是len(solution)
        # 比如solution=[0,1,1,4],则len(solution)=4,bucket[4]下标越界
        bucket = [0] * graph.node_num
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
                solution_pruned = solution[0:index1] + solution[index2:solution_len]
        self.solution = solution_pruned
        print 'after eliminate loop '
        print self.solution
        return self

    # 尽量往好的方向变异
    def find_path_new(self, graph, mutate_node):
        print '---------------enter into find_path_new-------'
        print 'mutate_node=',mutate_node
        solution_len = len(self.solution)
        # 变异结点的索引（下标）
        mutate_node_index = self.solution.index(mutate_node)
        previous_node = self.solution[mutate_node_index-1]
        candidate = []
        for node in range(graph.node_num):
            if (node != mutate_node and graph.get_connection_status(node, previous_node)) or (node == previous_node):
                candidate.append(node)
        if len(candidate) == 0:
            return self.solution
        current_node = choice(candidate)
        found = False
        tmp_solution = self.solution[0:mutate_node_index]
        if current_node == self.solution[solution_len-1]:
            tmp_solution.append(current_node)
            return tmp_solution
        node_adjs = graph.get_node_adjs()
        while not found:
            tmp_solution.append(current_node)
            for node in self.solution[-1:mutate_node_index:-1]:
                if graph.get_connection_status(node, current_node):
                    found = True
                    node_index = self.solution.index(node)
                    tmp_solution += self.solution[node_index:]
                    # tmp_solution.append(self.solution[solution_len - 1])
                    break
            if found:
                chromosome = Chromosome()
                chromosome.set_solution(tmp_solution)
                chromosome.calculate_fitness(graph, Population.delay_w)
                if chromosome.get_fitness() <= self.get_fitness():
                    return self.solution
                else:
                    return tmp_solution
            else:
                # 更新next_node,与next_node相邻而且没有访问过的结点
                current_node_adjs = []
                for node in node_adjs[current_node]:
                    print '--------enter into find path new-----'
                    if node > current_node:
                        current_node_adjs.append(node)
                if len(current_node_adjs) == 0:
                    return self.solution
                # 当前结点的相邻而未访问过的结点如果存在
                else:
                    current_node = choice(current_node_adjs)
        # default return value
        return self.solution

    # 根据graph和变异结点，返回新的解solution
    def find_path(self, graph, mutate_node):
        print '---------------enter into find_path-------'
        print 'mutate_node=',mutate_node
        node_adjs = graph.get_node_adjs()
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
        for node in range(graph.node_num):
            if node not in visited:
                tabu.append(node)
        for node in tabu:
            if graph.get_connection_status(node, self.solution[mutate_node_index-1]):
                candidate.append(node)
        print 'candidate=', candidate
        # 变异结点变异后变成的结点
        if len(candidate) == 0:
            return self.solution
        current_node = choice(candidate)
        found = False
        tmp_solution = self.solution[0:mutate_node_index]
        # while not found:
        while len(tabu) != 0:
            tmp_solution.append(current_node)
            visited.append(current_node)
            tabu.remove(current_node)
            for node in self.solution[-1:mutate_node_index:-1]:
                if graph.get_connection_status(node, current_node):
                    found = True
                    node_index = self.solution.index(node)
                    tmp_solution += self.solution[node_index:]
                    # tmp_solution.append(self.solution[solution_len - 1])
                    break
            if found:
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

    def mutate(self, graph, pm):
        print '----------------------------enter into Chromosome mutate-----------'
        random_p = random()
        if random_p > Global.pm:
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
        self.solution = self.find_path(graph, mutate_node)
        # self.solution = self.find_path_new(graph, mutate_node)
        self.solve_loop(graph)


# 种群
class Population:
    # DFS variation : 随机选择临接结点
    @staticmethod
    def random_chromosome(graph, src_node, des_node):
        print 'enter into random_chromosome'
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
                print '--------',node
                if graph.connection_status[current_node][node]:
                    print 'node=' ,node
                    print 'tabu=' ,tabu
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
                print 'next_node',next_node

                tabu.remove(next_node)
                visited.append(next_node)
                stack.append(current_node)
                current_node = next_node
                print 'find a adj node, current_node=',current_node
                if current_node == des_node:
                    flag = True
                    break
            # 没有相邻结点
            else:
                print 'cccccccc'
                print 'current_node=', current_node
                print 'visited=',visited
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

    delay_w = 15
    pc = 0.9
    pm = 0.1

    def __init__(self, graph, src_node, des_node, pop_scale, pc, pm, delay_w):
        self.pc = pc
        self.pm = pm
        self.delay_w = delay_w
        # 为何要定义一个graph，是因为用到的地方有多出，每次都从外面传进来，嫌麻烦，干脆自己定义一个成员变量算啦
        self.graph = graph
        # 种群规模
        self.pop_size = pop_scale
        # 个体（染色体）集合
        self.chromosomes = [Chromosome() for i in range(self.pop_size)]
        # 最佳适应度值
        self.best_fitness = 0
        # 平均适应度值
        self.avg_fitness = 0
        # 最好的解
        self.best_solution = []
        # 最优解对应的时延之和
        self.respective_delay = 0
        # 交配池
        self.mate_pool = []
        # 根据des_code逆向找到一条路径即为一个解
        for i in range(self.pop_size):
            self.chromosomes[i].set_solution(Population.random_chromosome(self.graph, src_node, des_node))

    def get_avg_fitness(self):
        return self.avg_fitness

    def get_popsize(self):
        return self.pop_size

    def get_best_fitness(self):
        return self.best_fitness

    # 计算适应度函数 F = ((Bw-B) + (D-Dw))/cost
    # 计算每一个个体的适应度值并记录最佳适应度值的个体
    # F = (C-cost) * {C1 - lg(Lw -[1-π(1 - LRi)])} * lg(Dw - D +10)
    def calculate_fitness(self):
        # 对种群中每个个体 计算适应度
        sum_fitness = 0
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness(self.graph, self.delay_w)
            sum_fitness += chromosome.get_fitness()
        # 计算适应度值的同时也计算了平均适应度值
        self.avg_fitness = sum_fitness / self.pop_size
        # 对种群按照适应度值进行从大到小排序
        self.chromosomes = sorted(self.chromosomes, key=attrgetter('fitness'),  reverse=True)
        # 更新种群最佳适应度值及其对应的解
        chromosome = self.chromosomes[0]
        if self.best_fitness < chromosome.get_fitness():
            self.best_fitness = chromosome.get_fitness()
            self.best_solution = chromosome.get_solution()
            delay_matrix = self.graph.get_delay()
            sum_delay = 0
            solution_len = len(self.best_solution)
            for i in range(solution_len-1):
                sum_delay += delay_matrix[self.best_solution[i]][self.best_solution[i+1]]
            self.respective_delay = sum_delay
        else:
            chromosome = Chromosome()
            chromosome.set_solution(self.best_solution)
            chromosome.set_fitness(self.best_fitness)
            self.chromosomes[self.pop_size-1] = chromosome


    def get_best_one_from_list(self, candidates=None):
        if candidates == None:
            return None
        max_fitness = 0
        index = -1
        for chromosome in candidates:
            index += 1
            fitness = chromosome.get_fitness()
            if fitness > max_fitness:
                max_fitness = fitness
                max_index = index
        return candidates[index]

    def choose_jbs(self):
        print '--------------------------enter into choose function----------------------------'
        print 'self.best_fitness=', self.best_fitness
        print 'self.best_solution=', self.best_solution
        best_chromosome = Chromosome()
        best_chromosome.set_solution(self.best_solution)
        best_chromosome.set_fitness(self.best_fitness)
        self.mate_pool = [best_chromosome]
        for num in range(self.pop_size - 1):
            candidates = sample(self.chromosomes,self.pop_size/2)
            self.mate_pool.append(self.get_best_one_from_list(candidates))
        self.mate_pool.sort()
        self.mate_pool.reverse()
        print 'len_mate_pool=', len(self.mate_pool)
        print 'mate_pool='
        for chrom in self.mate_pool:
            print chrom.get_solution()

    # 选择pop_size-1个个体，加上最佳适应度个体，放入交配池，为遗传操作做准备
    def choose(self):
        print '--------------------------enter into choose function----------------------------'
        print 'self.best_fitness=',self.best_fitness
        print 'self.best_solution=',self.best_solution
        best_chromosome = Chromosome()
        best_chromosome.set_solution(self.best_solution)
        best_chromosome.set_fitness(self.best_fitness)
        self.mate_pool = [best_chromosome]
        for num in range(self.pop_size-1):
            # 随机生成一个0到1之间的随机浮点数
            tmp_random = random()
            print 'tmp_random===',tmp_random
            sum_fitness = 0
            total_fitness = 0
            print 'self.pop_size=',self.pop_size
            for int_i in range(self.pop_size):
                print 'self.pop_size = ',self.pop_size
                print 'int_i =', int_i
                total_fitness += self.chromosomes[int_i].get_fitness()
            for int_i in range(self.pop_size):
                sum_fitness += self.chromosomes[int_i].get_fitness()*1.0/total_fitness
                print 'sum_fitness=',sum_fitness
                if tmp_random <= sum_fitness:
                    print 'tmp_random=',tmp_random,';sum_fitness=',sum_fitness
                    self.mate_pool.append(self.chromosomes[int_i])
                    break
        self.mate_pool.sort()
        self.mate_pool.reverse()
        print 'len_mate_pool=',len(self.mate_pool)
        print 'mate_pool='
        for chrom in self.mate_pool:
            print chrom.get_solution()

    # 交叉： 依照概率将交配池中的个体两两进行交叉操作
    # 寻找两条路径中相同的结点（非源结点与目的结点）
    def crossover(self):
        print '-----------------------------------------enter into crossover() of Population--------------------'
        print 'self.best_fitness=',self.best_fitness
        print 'self.best_solution=',self.best_solution
        mate_pool_bak = []
        print 'self.pop_size/2=', self.pop_size/2
        int_times= 0
        for num in range(self.pop_size/2):
            print '===========',int_times
            print 'before chromosome crossover'
            print 'self.best_fitness=', self.best_fitness
            print 'self.best_solution=', self.best_solution
            print 'chromosome one to be crossovered=',self.mate_pool[num*2].get_solution()
            print 'chromosome two to be crossovered=',self.mate_pool[num*2+1].get_solution()
            # mate_pool_bak += self.cross(self.mate_pool[num*2], self.mate_pool[num*2+1])
            mate_pool_bak += self.mate_pool[num*2].crossover_improv2(self.mate_pool[num*2+1], self.graph, self.pc)
            int_times += 1
        self.mate_pool = mate_pool_bak
        print 'mate_pool='
        for chromosome in self.mate_pool:
            print chromosome.get_solution()

    # 变异：依照概率对交配池中每个个体进行变异操作
    def mutate(self):
        print 'self.best_fitness=',self.best_fitness
        print 'self.best_solution=',self.best_solution

        for chromosome in self.mate_pool:
            chromosome.mutate(self.graph, self.pm)
            print 'mate_pool='
            for internal_chromosome in self.mate_pool:
                print internal_chromosome.get_solution()

    # 更新种群以及最佳适应值所对应的个体
    def update(self):
        print '-----------enter into update()------'
        print 'self.best_fitness=',self.best_fitness
        print 'self.best_solution=',self.best_solution
        self.chromosomes = sorted(self.mate_pool, key=attrgetter('fitness'),  reverse=True)
        new_best_fitness = self.chromosomes[0].get_fitness()
        print 'self.best_fitness:=', self.best_fitness
        print 'new_best_fitness = ', new_best_fitness
        if new_best_fitness > self.best_fitness:
            self.best_fitness = new_best_fitness
            # self.best_solution = self.best_chromosome.get_solution
            self.best_solution = self.chromosomes[0].get_solution()
        best_chromosome = Chromosome()
        best_chromosome.set_solution(self.best_solution)
        print 'min_cost=', best_chromosome.get_total_cost(self.graph)
        print 'self.chromosomes='
        for chrom in self.chromosomes:
            print chrom.get_solution()


if __name__ == '__main__':
    # 图形初始化
    starttime = clock()
    # f = open('test01.txt','r')
    # f = open('test02.txt', 'r')
    # f = open('test03.txt', 'r')
    # f = open('test03_new.txt', 'r')
    f = open('test04.txt', 'r')
    line = f.readline().split()
    print line
    node_num = int(line[0])
    edge_num = int(line[1])
    print node_num
    print edge_num
    graph = Graph(node_num, edge_num)
    f.readline()
    line = f.readline().split()
    src = int(line[0])
    dst = int(line[1])
    pop_scale = int(line[2])
    pc = float(line[3])
    pm = float(line[4])
    Global.pm = pm
    delay_w = int(line[5])
    Global.delay_w  = delay_w
    # (row_num, col_num, BandWidth, Delay, Cost)
    # param_length会随着的参数的增加而增大
    param_length = 5
    graph.init_edge_measure(f, param_length)
    graph.init_node_adjs()
    print '----------node_adjs----------'
    print graph.get_node_adjs()
    print '------------------->graph.cost<------------'
    print graph.cost
    # print graph.bandwidth[0][1]
    # print graph.cost[0][1]

    # 种群初始化
    # test01.txt
    # # 源点
    # src = 0
    # # 终点
    # dst = 4
    # # 种群规模
    # pop_scale = 6
    # # 交叉概率
    # pc = 0.99
    # # 变异概率
    # pm = 0.002
    # #时延约束
    # delay_w = 15

    # test02.txt
    # src = 0
    # dst = 5
    # pop_scale = 20
    # pc = 0.9
    # pm = 0.001
    # #时延约束
    # delay_w = 8
    population = Population(graph, src, dst, pop_scale, pc, pm, delay_w)
    pop_size = population.get_popsize()
    print 'pop_size=', pop_size

    generations = 0
    MAX_GENERATION = 100
    best_fitnesses = []
    avg_fitnesses = []
    min_costs = []
    flag = True
    count = 0
    TIME = 1000
    sum_generation = 0
    ratio = 0
    population.calculate_fitness()
    while generations < MAX_GENERATION:
        print '--------------------generations=>>>>>', generations, '<<<<--------------'
        # 计算种群中所以个体的适应度值
        # population.calculate_fitness()
        for i in range(pop_size):
            # s1 = Population.random_chromosome(graph, 0, 4)
            s1 = population.chromosomes[i]
            print 'i=', i, ': ', s1.get_solution(), ";Fitness=%.6f" % (s1.get_fitness())
        population.choose()
        # population.choose_jbs()
        population.crossover()
        population.mutate()
        population.update()
        population.calculate_fitness()
        avg_fitness = population.avg_fitness
        avg_fitnesses.append(avg_fitness)
        best_fitnesses.append(population.get_best_fitness())
        best_chromosome = Chromosome()
        best_chromosome.set_solution(population.best_solution)
        min_cost = best_chromosome.get_total_cost(graph)
        min_costs.append(min_cost)
        # if flag and fabs(population.get_best_fitness()*100-2.77777777778) >= 1.0e-11:
        #     sum_generation += generations
        #     flag = False
        # 自适应变异概率,随着种群的平均适应度值变大，其变异概率应该减小
        # Global.pm = 1 - population.avg_fitness/population.best_fitness
        generations += 1
    # long running
    endtime = clock()
    # 只计算程序运行的CPU时间
    print "program costs time is %.8f s" % (endtime - starttime)
    x = [i for i in range(MAX_GENERATION)]
    y = best_fitnesses
    z = avg_fitnesses
    u = min_costs
    info = 'node_num=%d, edge_num=%d, pop_scale=%d, r=%.3f pc=%.3f, pm=%.3f, global_min_cost=%d, best_solution=%s, respective_delay=%d'
    value = (node_num, edge_num, pop_scale,Global.r, pc, pm, min_cost, population.best_solution, population.respective_delay)
    pl.figure(info % value)
    pl.subplot(211)
    # pl.xlabel('generation')
    pl.ylabel('fitness')
    pl.plot(x, y, 'r-', label='best_fitness')
    pl.plot(x, z, 'b.-', label='avg_fitness')
    # pl.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    # 将标注放在右下角
    pl.legend(loc='lower right')
    pl.subplot(212)
    pl.xlabel('generation')
    pl.ylabel('cost')
    pl.plot(x, u, 'r.-', label='min_cost')
    pl.legend()
    # pl.text(90, 4, '--min_cost', color='red')


    pl.show()
#qq
