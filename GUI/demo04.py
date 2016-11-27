import numpy as np
import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5]  # Make an array of x values
# y = [1, 4, 9, 16, 25]  # Make an array of y values for each x value
x= np.arange(0,100,1)
y = [s/3.0 for s in x]
plt.plot(x,y,'b--')
plt.ylabel('y axes')
plt.xlabel('x axes')
plt.show()
