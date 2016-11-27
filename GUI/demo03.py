# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 100, 100)
plt.figure('one')  # ❶ # 选择图表1
# plt.plot(x, np.exp(x / 3))
# plt.plot(x, 100/ np.power(2, x), )
# plt.plot(x, x/2)
y = np.exp(x/3)
plt.plot(x, y)

plt.show()
