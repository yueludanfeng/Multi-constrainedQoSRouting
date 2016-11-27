# coding=utf-8
# Dijkstra算法，求两点之间最短距离和相应的路径,注意coding与utf-8之间不能有空格
import sys
import heapq


class Vertex:
    def __init__(self, node):
        self.id = node

        # adjacent用于存储该结点(相邻边)出边
        self.adjacent = {}

        # Set distance to infinity for all nodes, which means cannot be visited
        # 从源点开始到该结点为止，权值之和， sum of weight,
        self.distance = sys.maxint

        # Mark all nodes unvisited 该结点是否已经访问
        self.visited = False

        # Predecessor 该结点的前驱结点
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def get_visited(self):
        return self.visited

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])


class Graph:

    def __init__(self,visit_path=[]):
        # 结点集合
        self.vert_dict = {}
        # 结点数目
        self.num_vertices = 0
        self.visit_path=[]
        self.stop_flag = False

    def __iter__(self):
        return iter(self.vert_dict.values())

    # 添加结点
    def add_vertex(self, node):
        self.num_vertices += 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    # 获取结点
    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    # 添加边
    def add_edge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost) # 说明是无向图
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    # 获取结点集
    def get_vertices(self):
        return self.vert_dict.keys()

    def dfs(self, src_node, dest_node):
        # print dest_node.get_id()
        if self.stop_flag is True:
            return
        for node in src_node.get_connections():
            if node.get_visited() is True:
                continue
            else:
                print 'node.visited',node.get_visited()
                self.visit_path.append(node.get_id())
                node.set_visited()
                print 'node=', node.get_id()
                print 'dest_node=', dest_node.get_id()
                if node.get_id() == dest_node.get_id():
                    print '='
                    self.stop_flag = True
                    return
                else:
                    self.dfs(node, dest_node)


# 给定一个结点，获取以该节点为终点的路径所包含的结点集合
def shortest(target_node, path_set):
    if target_node.previous:
        path_set.append(target_node.previous.get_id())
        shortest(target_node.previous, path_set)
    return


def dijkstra(aGraph, start, target):
    print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(), v) for v in aGraph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)

            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
               # print 'updated : current = %s next = %s new_dist = %s' % (current.get_id(), next.get_id(), next.get_distance())
#            else:
               # print 'not updated : current = %s next = %s new_dist = %s' % (current.get_id(), next.get_id(), next.get_distance())

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(), v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


if __name__ == '__main__':

    visitpath = ['a']
    g = Graph(visitpath)

    g.add_vertex('a')
    g.add_vertex('b')
    g.add_vertex('c')
    g.add_vertex('d')
    g.add_vertex('e')
    g.add_vertex('f')

    g.add_edge('a', 'b', 7)
    g.add_edge('a', 'c', 9)
    g.add_edge('a', 'f', 14)
    g.add_edge('b', 'c', 10)
    g.add_edge('b', 'd', 15)
    g.add_edge('c', 'd', 11)
    g.add_edge('c', 'f', 2)
    g.add_edge('d', 'e', 6)
    g.add_edge('e', 'f', 9)

    print 'Graph data:'
    for v in g:
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            print '( %s , %s, %3d)' % (vid, wid, v.get_weight(w))

    g.get_vertex('a').set_visited()

    visit = g.dfs(g.get_vertex('a'), g.get_vertex('e'))
    print 'visipath=',g.visit_path

    # dijkstra(g, g.get_vertex('a'), g.get_vertex('e'))
    #
    # target = g.get_vertex('e')
    # path = [target.get_id()]
    # shortest(target, path)
    # print 'The shortest path : %s' % (path[::-1])

