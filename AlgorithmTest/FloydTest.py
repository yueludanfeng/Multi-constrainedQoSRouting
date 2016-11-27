# coding=utf-8
# Floyd求两点之间最短距离和对应的路径
# 参考网址：http://wiki.jikexueyuan.com/project/easy-learn-algorithm/floyd.html


INFINITY = 1000


class Graph:
    def __init__(self, vertexNum, edgeNum):
        self.vertextNum = vertexNum
        self.edgeNum = edgeNum
        # 二维数组的定义 参考：http://www.cnblogs.com/btchenguang/archive/2012/01/30/2332479.html
        self.e = [([0] * self.vertextNum) for i in range(self.vertextNum)]

    # 添加边
    def addEdge(self, start, end, distance):
        self.e[start][end] = distance

    def find_shortest_path(self):
        for k in range(self.vertextNum):
            for i in range(self.vertextNum):
                for j in range(self.vertextNum):
                    if self.e[i][j] > self.e[i][k] + self.e[k][j]:
                        self.e[i][j] = self.e[i][k]+self.e[k][j]

    def print_distance(self):
        for i in range(self.vertextNum):
            for j in range(self.vertextNum):
                print '(%d,%d)=%d' % (i, j, self.e[i][j])
            print

    def get_shortest_distance(self, src, dst):
        return self.e[src][dst]

    def get_shortest_path(self, src, dst):
        path = []
        while dst != src:
            path.append(dst)
            # print path
            min_value = INFINITY
            min_value_index = -1
            for i in range(self.vertextNum):
                if i == dst:
                    continue
                if min_value > self.e[i][dst]:
                    min_value = self.e[i][dst]
                    min_value_index = i

            dst = min_value_index
        path.append(src)
        return path


class GraphD:
    def __init__(self, vertext_num, edge_num):
        self.vertext_num = vertext_num
        self.edge_num = edge_num
        self.e = [([0] * self.vertext_num) for i in range(self.vertext_num)]

    # 添加边
    def addEdge(self, start, end, distance):
        self.e[start][end] = distance

    def find_shortest_path(self):
        for index in range(self.vertext_num):
            dp = [i for i in self.e[index]]
            visited = [False for i in range(self.vertext_num)]
            visited[index] = True
            for times in range(self.vertext_num-1):
                #找当前未访问的结点中距离index最近的那个
                min_value = INFINITY
                min_value_index = -1
                for i in range(self.vertext_num):
                    if (visited[i] == False ) and(dp[i] < min_value):
                        min_value = dp[i]
                        min_value_index = i
                visited[min_value_index] = True
                #松弛
                for i in range(self.vertext_num):
                    if self.e[min_value_index][i]>0 and self.e[min_value_index][i]+dp[min_value_index]<dp[i]:
                        dp[i] = self.e[min_value_index][i]+dp[min_value_index]
                        self.e[index][i] = dp[i]

            list = []
            for key in self.e[index]:
                list.append(key)
            # for key in dp:
            #     list.append(key)
                # print self.e[index][i]
            print list

    def print_distance(self):
        for i in range(self.vertext_num):
            for j in range(self.vertext_num):
                print '(%d,%d)=%d' % (i, j, self.e[i][j])
            print

    def get_shortest_path(self, src, dst):
        path = []
        while dst != src:
            path.append(dst)
            # print path
            min_value = INFINITY
            min_value_index = -1
            for i in range(self.vertext_num):
                if i == dst:
                    continue
                if min_value > self.e[i][dst]:
                    min_value = self.e[i][dst]
                    min_value_index = i

            dst = min_value_index
        path.append(src)
        return path

if __name__ == '__main__':
    # 初始化,给定结点与边
    G = GraphD(4,8)
    G.addEdge(0, 1, 2)
    G.addEdge(0, 2, 6)
    G.addEdge(0, 3, 4)

    G.addEdge(1, 0, INFINITY)
    G.addEdge(1, 2, 3)
    G.addEdge(1, 3, INFINITY)

    G.addEdge(2, 0, 7)
    G.addEdge(2, 1, INFINITY)
    G.addEdge(2, 3, 1)

    G.addEdge(3, 0, 5)
    G.addEdge(3, 1, INFINITY)
    G.addEdge(3, 2, 12)

    G.print_distance()
    # 寻最短路径
    G.find_shortest_path()

    # 打印
    G.print_distance()

    # print 'distance(%s,%s) = %d' % (0, 1, G.get_shortest_distance(0, 1))

    path = G.get_shortest_path(1, 3)
    path.reverse()
    print 'path=',path




