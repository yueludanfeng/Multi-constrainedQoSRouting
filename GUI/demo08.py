import math
import matplotlib.pyplot as plt


plt.subplot(311)
plt.plot([1,2,3], label="test1")
plt.plot([3,2,1], label="test2")
# Place a legend above this legend, expanding itself to
# fully use the given bounding box.
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(323)
plt.plot([1,2,3], label="test1")
plt.plot([3,2,1], label="test2")
# Place a legend to the right of this smaller figure.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(313)
x = [1] * 5
print 'hello'
y = [i for i in range(5)]
plt.plot(x,y,'r-*')
plt.show()
print math.abs(-3)
plt.figure
