# coding=utf-8
# Author:   Lixinming
"""
功能：计算算法获得最佳解的成功率,并写入与其文件名相应的final文件中
"""
from os import listdir
from os.path import isfile
from os.path import basename
mypath = "./"
# LOOP_TIME = 10.0
onlyfiles_contains_result = [f for f in listdir(mypath) if isfile(f) and basename(f).__contains__("rate") ]
for file in onlyfiles_contains_result:
    #print file
    fp = open(file,"r")
    sum = 0
    for line in fp:
        sum += 1
    # sum /= LOOP_TIME
    index_of_underline = file.index("_")
    appendix = file[index_of_underline:len(file)]
    file_name = "final".__add__(appendix)
    print file_name
    fiter_time = open(file_name, "a+")
    fiter_time.write(str(sum).__add__("\n"))
