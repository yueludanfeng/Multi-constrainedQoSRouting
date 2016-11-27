import matplotlib.pyplot as plt

X1 = range(0, 50, 100)
Y1 = [num ** 2 for num in X1]  # y = x^2 X2 = [0, 1] Y2 = [0, 1] # y = x
Y1 = [i/2 for i in X1]
plt.figure('five')
Fig = plt.figure(figsize=(8,4))  # Create a `figure' instance # Create a `axes' instance in the figure
# Ax = Fig.add_subplot(111)
plt.plot(X1, Y1) # Create a Line2D instance in the axes
plt.show()


# Fig.savefig("test.pdf")