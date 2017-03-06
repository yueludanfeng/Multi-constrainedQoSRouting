# coding=utf-8
# Author:   Lixinming

from os import listdir
from os.path import isfile
from os.path import basename
from os import system
from sys import argv
mypath = "./"
onlyfiles_contains_result = [f for f in listdir(mypath) if isfile(f) and basename(f).__contains__("final") ]
file_total = open("total_sum","a+")
for file in onlyfiles_contains_result:
    # truncate later spice
    index_of_underline = file.index("_")
    appendix = file[index_of_underline+1:len(file)]
    #print appendix
    values = appendix.split("_")
    #print values
    # read contents from file
    fp = open(file, "r")
    for line in fp:
        value = float(line)
        values.append(value)
    for value in values:
        file_total.write(str(value).__add__("\t"))
    file_total.write("\n")
system("sort -u total_sum >> ./result/test")

