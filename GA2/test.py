# coding=utf-8
from math import log10
from math import exp
from random import randint
from random import choice
# 获取当前时间
import time
a = time.localtime(time.time())
s = time.strftime('%Y-%m-%d %H:%M:%S', a)
print s

s1 = [1, 2, 3, 4]
s2 = [2, 4]
res = []
for i in s1:
    if i not in s2:
        res.append(i)
print res

d = {}
d['a'] =1
d[1]=[1,2,3]
for k in d:
    print k,'=>',d[k]

rec = {r:[] for r in range(2)}
m = [[1,2,3],[4,5,6]]
for i in range(2):
    for j in range(3):
        if m[i][j]>2:
            rec[i].append(m[i][j])
print rec

a = [0,1,2,3,4,5]
print a[5:1:-1]
print a[-1:0:-1]
print a[2:-1]+[a.index(len(a)-1)]
a += [10.12]
print a
dict = {}
dict[1] = [1,2,3]
dict[2] = [4,5,6]
rand = randint(1,2)
print len(dict[rand])
for node in dict:
    print dict[node]
x = set('spam')
print x

s = set()
s.add(1)
s.add(2)

a = 1 * exp(-1*24) * exp(200)
b = 1 * exp(-1*25) *1000 *exp(200)

print a
print b
print a-b

a1=log10(3)
