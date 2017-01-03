# coding=utf-8
# Author:   Lixinming

import os
print os.getcwd()
# os.system("python ACO_bak.py")
# os.system("ls")
os.chdir("../ACO")
print os.getcwd()
for i in range(100):
    os.system("python ACO_bak.py")