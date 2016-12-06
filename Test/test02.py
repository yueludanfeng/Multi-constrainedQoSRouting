# coding=utf-8
# Author:   Lixinming

from random import random
class A:
    a = 1
    b = 2
    def __init__(self):
        print A.a
A()

from math import pow

a = pow(2,3)
print 4/a

dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
dict = sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
print dict
print type(dict)
dict = sorted(dic.items(), key=lambda d:d[1], reverse = True)
print dict
print type(dict)
print dict[0]
print dict[0][0]
print dict[0][1]
print 'random of 0 and 1 is ',random()

res = [(1, 0.5), (3, 0.3888888888888889), (2, 0.1111111111111111)]
for k, v in res:
    print k, '--', v
    print type(k),type(v)

print list(res)
a = [1, 2]
print 'a.pop()=',a.pop()
print 'a.pop()=',a.pop()

print 'he'
while True:
    for i in range(5):
        print i
    if i == 4:
        print 'a'
        break
    else:
        print 'hh'

lista = [1,2,3]
lena = len(lista)
print 'end is',lista[lena-1]
def add(li):
    li.append(1)
a = []
add(a)
print 'a=',a
example = [1,2,3]
print example[::-1]
print 1+0.0
print "%.1f" % (1/2.0)