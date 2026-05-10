import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    return np.log(1 + np.exp(x))

plt.figure()
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, tanh(x))
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, relu(x))
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, leaky_relu(x))
plt.title("Leaky ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, elu(x))
plt.title("ELU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

plt.figure()
plt.plot(x, softplus(x))
plt.title("Softplus Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()




# def mcp_andnot(x1, x2):
#     w1 = 1
#     w2 = -1
#     threshold = 1
#     net = (x1 * w1) + (x2 * w2)
#     if net >= threshold:
#         return 1
#     else:
#         return 0

# print("X1 X2 Output")
# for x1 in [0, 1]:
#     for x2 in [0, 1]:
#         print(x1, x2, mcp_andnot(x1, x2))