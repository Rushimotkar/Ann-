import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    return x * (1 - x)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

wh = np.random.rand(2, 3)
bh = np.random.rand(1, 3)
wo = np.random.rand(3, 1)
bo = np.random.rand(1, 1)
lr = 0.1

for i in range(10000):
    hidden = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hidden, wo) + bo)
    error = y - output
    d_output = error * derivative(output)
    d_hidden = d_output.dot(wo.T) * derivative(hidden)
    wo += hidden.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

print("Predicted Output:")
print(output)
