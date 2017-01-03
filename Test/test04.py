# coding=utf-8
# Author:   Lixinming
import os
os.chdir("../ACO")
result = 0
# for line in open("result.txt","r"):
#     # result += int(line)
#     # print type(line)
#     print line
# print result
str = ""
linenum = 0
file = open("result.txt","r")
for line in file:
    str+=line
    linenum+=1
print str
s = str.split("\n")
for e in s:
    if e!="":
        result += float(e)
print result
print linenum
