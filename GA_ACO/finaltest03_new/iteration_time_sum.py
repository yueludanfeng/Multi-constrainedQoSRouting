# coding=utf-8
# Author:   Lixinming
"""
功能：计算算法获得最优解的最小迭代次数,并写入与其文件名相应的final文件中
"""
from os import listdir
from os.path import isfile
from os.path import basename
# LOOP_TIME = 1000.0
LOOP_TIME = 100.0
mypath = "./"
onlyfiles_contains_result = [f for f in listdir(mypath) if isfile(f) and basename(f).__contains__("iter_time") ]
for file in onlyfiles_contains_result:
    fp = open(file,"r")
    sum = 0
    for line in fp:
        sum += float(line)
    iter_time = sum / LOOP_TIME

    first_index_of_underline = file.index("_")
    index_of_underline = file.index("_", first_index_of_underline+1)
    appendix = file[index_of_underline:len(file)]
    file_name = "final".__add__(appendix)
    print file_name
    fiter_time = open(file_name, "a+")
    fiter_time.write(str(iter_time))
