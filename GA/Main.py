# coding=utf-8
# Author:Lixinming
from sys import argv
from sys import exit
from os import system

if __name__ == "__main__":
    """
    输入: python Main.py new 或者 python Main.py old
    相应地执行ga_statistics_with_calculate_fitness_new.py或ga_statistic_with_calculate_fitness_old.py
    """
    calculate_function_type = ""
    if len(argv) == 2:
        calculate_function_type = argv[1]
    else:
        exit(0)
    if calculate_function_type == "new":
        print "new"
        # system("python ga_statistic_with_calculate_fitness_new.py")
    elif calculate_function_type =="old":
        print "old"
        # system("python ga_statistic_with_calculate_fitness_old.py")
    else:
        pass