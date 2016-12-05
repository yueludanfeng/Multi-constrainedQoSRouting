# coding=utf-8
from math import pow
from random import randrange
from random import random
class Graph:
    def __init__(self, node_num, edge_num):
        # 结点数
        self.node_num = node_num
        # 边数
        self.edge_num = edge_num
        # 连接状态
        self.connection_status = [[False for col in range(self.node_num)] for row in range(self.node_num)]
        # 带宽
        self.bandwidth = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        # 时延
        self.delay = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        # 花费
        self.cost = [[0 for col in range(self.node_num)] for row in range(self.node_num)]
        # 邻接点集合
        self.node_adjs = {r:[] for r in range(self.node_num)}
        # 总花费
        self.total_cost = 0
        # 总时延
        self.total_delay = 0

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

    def add_bandwidth(self, rownum, colnum, value=0):
        self.bandwidth[rownum][colnum] = value
        self.bandwidth[colnum][rownum] = value

    def add_cost(self, rownum, colnum, value=0):
        self.cost[rownum][colnum] = value
        self.cost[colnum][rownum] = value
        self.total_cost += value

    def add_delay(self, rownum, colnum, value=0):
        self.delay[rownum][colnum] = value
        self.delay[colnum][rownum] = value
        self.total_delay += value

    def add_edge_measure(self, rownum, colnum, bandwidth, delay, cost):
        self.add_bandwidth(rownum, colnum, bandwidth)
        self.add_delay(rownum, colnum, delay)
        self.add_cost(rownum, colnum, cost)
        self.add_connection_status(rownum, colnum)

    # 初始化各边的度量参数值
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

    # 记录每个结点的相邻结点  like as follows:     rec = {1:[2,3],  2:[3,4]} and so on
    def init_node_adjs(self):
        for row_num in range(self.node_num):
            for col_num in range(self.node_num):
                if self.connection_status[row_num][col_num]:
                    self.node_adjs[row_num].append(col_num)
        print self.node_adjs


class GlobalInfo:
    alpha = 1
    beta = 1
    rho = 0.1
    # 如果随机数小于或等于r则选择max{pher(r,s)}
    # 否则按照概率公式进行转移
    r = 0.2
    # pheromone[][] 全局信息素初始值
    C = 1
    node_num = 0
    edge_num = 0
    src_node = 0
    dst_node = 0
    ant_num = 0
    delay_w = 0
    pheromone = None
    delta_pheromone = None

    @staticmethod
    def init_param(node_num, edge_num, src_node, dst_node, ant_num, delay_w):
        GlobalInfo.node_num = node_num
        GlobalInfo.edge_num = edge_num
        GlobalInfo.src_node = src_node
        GlobalInfo.dst_node = dst_node
        GlobalInfo.ant_num = ant_num
        GlobalInfo.delay_w = delay_w


    def __init__(self):
        # 全局信息素矩阵初始`化
        GlobalInfo.pheromone = [[GlobalInfo.C for col in range(GlobalInfo.node_num)] for row in range(GlobalInfo.node_num)]
        # 全局信息素变化矩阵
        GlobalInfo.delta_pheromone = [[0 for col in range(GlobalInfo.node_num)] for row in range(GlobalInfo.node_num)]
        print "GlobalInfo.pheromone=",GlobalInfo.pheromone
        print "GlobalInfo.delta_pheromone=",GlobalInfo.delta_pheromone


class Ant:

    # 初始化
    def __init__(self, src_node, graph):
        self.graph = graph
        city_list = [i for i in range(GlobalInfo.node_num)]
        # 允许访问的城市(结点)集合
        self.allowed = city_list
        # 初始城市
        self.first_city = src_node
        # 当前城市
        self.current_city = self.first_city
        # 下一个城市
        self.next_city = -1
        # 禁忌表(不允许访问的城市（结点）)
        self.tabu = [self.first_city]
        # 允许访问的城市
        self.allowed.remove(self.first_city)
        # 解
        self.solution = [self.first_city]
        # 用于回溯
        self.Stack = []
        self.Stack.append(self.current_city)
        # delta[][] 数组初始化
        self.delta_pheromone = [[0 for col in range(GlobalInfo.node_num)] for row in range(graph.node_num)]

    # 选择下一个城市（结点）
    def choose_next_city(self):
        # if rand_value > r, 按照此公式(依照概率，按照赌轮选择)计算；否则选择pher(r,s)最大的结点
        print '-------------enter into choose_next_city()-----'
        # 计算路径迁移概率
        transe_p = {}
        current_city = self.current_city
        print 'current_city = ', current_city
        # 计算分母
        fm_sum = 0
        print 'self.allowed =', self.allowed
        count = 0
        for index in self.allowed:
            # 在允许访问的结点中只需要考虑相邻结点（城市）
            if not self.graph.get_connection_status(current_city, index):
                count += 1
                print 'count = ', count
                continue
            fm_sum += pow(GlobalInfo.pheromone[self.current_city][index], GlobalInfo.alpha) * \
                      pow(self.graph.get_cost()[self.current_city][index], GlobalInfo.beta)
            print 'fm_sum = ', fm_sum
        # 如果分母为0，则说明当前结点没有相邻结点,需要回溯
        if fm_sum == 0:
            print 'fm_sum==0'
            # 表示没有找到下一个结点，而且赋值之后，要直接令返回，否则后面的代码会继续执行
            self.next_city = -1
            return

        for city_index in self.allowed:
            # 在允许访问的结点中只需要考虑相邻结点（城市)
            if not self.graph.get_connection_status(current_city, city_index):
                continue
            # 计算分子
            print 'aaaaaaa'
            fz = pow(GlobalInfo.pheromone[self.current_city][city_index], GlobalInfo.alpha) * (pow(self.graph.get_cost()[self.current_city][city_index], GlobalInfo.beta))
            print '-----fz = ', fz
            transe_p[city_index] = fz / fm_sum
        # 获得概率最大的
        res = sorted(transe_p.items(), key=lambda d: d[1], reverse=True)
        # 获取最大概率对应的结点(城市)编号
        print "res =", res
        rand_value = random()
        print 'rand_value=',rand_value
        # 如果随机数不大于r,则选择概率最大的；否则以概率选择(有的论文排了序，有的没有排序)
        if rand_value <= GlobalInfo.r:
            print 'rand_value <= GlobalInfo.r:'
            self.next_city = res[0][0]
        rand_prob = random()
        print 'rand_prob=',rand_prob
        sum_prob = 0
        for k, v in res:
            sum_prob += v
            print 'sum_prob=',sum_prob
            if sum_prob >= rand_prob:
                self.next_city = k
                print 'self.next_city=',self.next_city
                break

    def move_to_next_city(self):
        # 没找到下一个结点
        if self.next_city == -1:
            self.solution.pop()
            top = self.Stack.pop()
            if top == self.current_city:
                self.current_city = self.Stack.pop()
            else:
                self.current_city = top
        # 找到下一个结点
        else:
            self.current_city = self.next_city
            self.next_city = -1
            # self.solution.append(self.current_city)
            self.allowed.remove(self.current_city)
            self.tabu.append(self.current_city)
            self.solution.append(self.current_city)
            self.Stack.append(self.current_city)
    # 局部信息素的更新，只更新该蚂蚁走过的路径上的信息素
    # 是每走一步更新一次，还是等走完之后在更新信息素呢
    # 答案是：每走一步都要更新一次；走完之后还要更新一次
    # Tao(i, j)<t+1> = (1-rho1) * Tao(i, j)<t> + delta_tao(i, j)
    # 参考《遗传算法融合蚁群算法》
    # 局部信息素更新

    def update_pheromone(self):
        pass
        # GlobalInfo.pheromone[self.current_city][self.next_city] = (1-GlobalInfo.q0) * GlobalInfo.pheromone[self.current_city][self.next_city] + \
        #                                                          GlobalInfo.q0 * 1.0 / (self.graph.get_delay()[self.current_city][self.next_city])

    def solve(self):
        # while self.current_city != GlobalInfo.dst_node:
        while True:
            self.choose_next_city()
            self.move_to_next_city()
            if self.current_city == GlobalInfo.dst_node:
                break
            else:
                print 'current_city=',self.current_city
        # 万万没想到,Stack就是解
        # self.solution = self.Stack
        self.update_pheromone()
        print 'self.solution = ',self.solution


class Population:
    def __init__(self, ):
        self.ant_num = GlobalInfo.ant_num
        self.ants = [Ant(GlobalInfo.src_node, obj_graph) for num in range(self.ant_num)]

    def solve(self):
        for ant in self.ants:
            ant.solve()



if __name__ == "__main__":
    # 读取文件中相关信息
    fp = open("test03.txt")
    line = fp.readline().split()
    node_num = int(line[0])
    edge_num = int(line[1])
    fp.readline()
    line = fp.readline().split()
    src_node = int(line[0])
    dst_node = int(line[1])
    ant_num = int(line[2])
    delay_w = int(line[3])

    print node_num
    print edge_num
    print ant_num
    print delay_w
    GlobalInfo()
    GlobalInfo.init_param(node_num, edge_num, src_node, dst_node, ant_num, delay_w)
    obj_graph = Graph(node_num, edge_num)
    param_length = 5
    obj_graph.init_edge_measure(fp, param_length)
    obj_graph.init_node_adjs()
    print '----------node_adjs----------'
    print obj_graph.get_node_adjs()
    print '------------------->graph.cost<------------'
    print obj_graph.cost

    population = Population()
    population.solve()




