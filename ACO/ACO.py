from math import pow
from random import randrange


class Ant:

    # 初始化
    def __init__(self, city_num=0):
        city_list = [i for i in range(city_num)]
        # 城市数目
        self.city_num = city_num
        # 初始城市
        self.first_city = randrange(city_list)
        # 当前城市
        self.current_city = self.first_city
        # 禁忌表
        self.tabu = [self.first_city]
        # 允许访问的城市
        self.allowed = city_list.reverse(self.first_city)
        # 访问的路径和
        self.tour_length = 0
        # delta[][] 数组初始化
        self.delta = [[0 for col in range(self.city_num)] for row in range(self.city_num)]

    # 选择下一个城市（结点）
    def choose_next_city(self, distance=[[]], pheromone=[[]], alpha=1, beta=1, graph=[[]], edge_num=0):

        # 计算路径迁移概率
        transe_p = [0 for i in range(self.city_num)]

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






