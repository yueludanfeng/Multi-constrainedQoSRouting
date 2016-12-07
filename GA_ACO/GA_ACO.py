# coding=utf-8
# Author:   Lixinming
# 融合了遗传算法和蚁群算法的混合算法


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

    def set_connection_status(self, row, col):
        self.connection_status[row][col] = True
        self.connection_status[col][row] = True

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
            self.add_edge_measure(*param)

    # 记录每个结点的相邻结点  like as follows:     rec = {1:[2,3],  2:[3,4]} and so on
    def init_node_adjs(self):
        for row_num in range(self.node_num):
            for col_num in range(self.node_num):
                if self.connection_status[row_num][col_num]:
                    self.node_adjs[row_num].append(col_num)
        print self.node_adjs


