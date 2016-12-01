# coding=utf-8
from math import pow
from random import randrange


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
    q0 = 0.2
    q1 = 0.2
    node_num = 0
    edge_num = 0
    src_node = 0
    dst_node = 0
    ant_num = 0
    delay_w = 0
    C =1
    # 全局信息素矩阵初始化
    pheromone = [[C for col in range(node_num)] for row in range(node_num)]
    # 全局信息素变化矩阵
    delta_pheromone = [[0 for col in range(node_num)] for row in range(node_num)]


    def __init__(self):
        pass


class Ant:

    # 初始化
    def __init__(self, src_node, graph):
        city_list = [i for i in range(graph.node_num)]
        self.allowed = city_list
        # 城市数目
        # 初始城市
        self.first_city = src_node
        # 当前城市
        self.current_city = self.first_city
        # 禁忌表(不允许访问的城市（结点）)
        self.tabu = [self.first_city]
        # 允许访问的城市
        self.allowed.remove(self.first_city)
        # 访问的路径和
        self.tour_length = 0
        # delta[][] 数组初始化
        self.delta = [[0 for col in range(graph.node_num)] for row in range(graph.node_num)]

    # 选择下一个城市（结点）
    def choose_next_city(self):

        # 计算路径迁移概率
        transe_p = [0 for i in range(GlobalInfo.node_num)]

        for city_index in self.allowed:
            # [BandWidth Delay Jitter] Delay equals 0 denotes disconnection
            if graph[self.current_city][city_index][1] == 0:
                continue
            # 计算分子
            fz = pow(pheromone[self.current_city][city_index], alpha)*(pow(distance[self.current_city][city_index], beta))
            # 计算分母
            fm_sum = 0
            for index in self.allowed:
                fm_sum += pow(pheromone[self.current_city][index], alpha) * pow(pheromone[self.current_city][index], beta)
            transe_p[i] = fz/fm_sum
        # 选择合适的


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
    GlobalInfo.node_num = node_num
    GlobalInfo.edge_num = edge_num
    GlobalInfo.src_node = src_node
    GlobalInfo.dst_node = dst_node
    GlobalInfo.ant_num  = ant_num
    GlobalInfo.delay_w = delay_w
    obj_graph = Graph(node_num, edge_num)
    param_length = 5
    obj_graph.init_edge_measure(fp, param_length)
    obj_graph.init_node_adjs()
    print '----------node_adjs----------'
    print obj_graph.get_node_adjs()
    print '------------------->graph.cost<------------'
    print obj_graph.cost






