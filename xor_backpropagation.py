import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

wh = np.random.uniform(size=(2,2))
bh = np.random.uniform(size=(1,2))
wo = np.random.uniform(size=(2,1))
bo = np.random.uniform(size=(1,1))
lr = 0.1

for epoch in range(10000):
    hidden = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hidden, wo) + bo)
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(wo.T) * sigmoid_derivative(hidden)
    wo += hidden.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

print("Final Output:\n", output)
