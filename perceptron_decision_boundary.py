import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2, 3], [1, 1], [-1, -2], [-2, -3]])
y = np.array([1, 1, -1, -1])
w = np.zeros(2)
b = 0
lr = 0.1

for epoch in range(10):
    for i in range(len(X)):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w = w + lr * y[i] * X[i]
            b = b + lr * y[i]

print("Weights:", w)
print("Bias:", b)

plt.scatter(X[:, 0], X[:, 1], c=y)
x_vals = np.linspace(-3, 3, 100)
if w[1] != 0:
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Boundary")
plt.show()
