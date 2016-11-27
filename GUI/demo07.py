#_*_coding:utf8
import numpy as np
import pylab as pl

x1 = [1, 2, 3, 4, 5, 6, 7]# Make x, y arrays for each graph
y1 = [456.4, 412.98, 378.34, 345.98, 310.3, 297.89, 234.9]

x2 = [1, 2, 3, 4, 5, 6, 7]
y2 = [478.45, 450.9, 411.89, 369.6, 340.34, 305.67, 286.56]
# use pylab to plot x and y
pl.plot(x1, y1, 'r')
pl.plot(x2, y2, 'g')

pl.title('LevelDB - DA-LSM')# give plot a title
pl.xlabel('recordcount axis') # make axis labels
pl.ylabel('throughput axis')

pl.xlim(0.0, 9.0)# set axis limits
pl.ylim(200.0, 500.0)

pl.show()# show the plot on the screen