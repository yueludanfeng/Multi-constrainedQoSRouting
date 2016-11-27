from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi)
offsets = np.linspace(0, 2*np.pi, 1, endpoint=False)
# Create array with shifted-sine curve along each column
yy = np.transpose([np.sin(x + phi) for phi in offsets])

plt.rc('lines', linewidth=3)
plt.rc('axes', prop_cycle=(cycler('color', [ 'g']) +
                           cycler('linestyle', [ '--'])))
fig, ax0 = plt.subplots(nrows=1)
ax0.plot(yy)
ax0.set_title('Set default color cycle to rgby')

# ax1.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
#                    cycler('lw', [1, 2, 3, 4]))
# ax1.plot(yy)
# ax1.set_title('Set axes color cycle to cmyk')

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.3)
plt.show()