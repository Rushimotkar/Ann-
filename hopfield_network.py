import numpy as np

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

n = patterns.shape[1]
W = np.zeros((n, n))

for p in patterns:
    p = p.reshape(-1, 1)
    W += p @ p.T

np.fill_diagonal(W, 0)

print("Weight Matrix:\n", W)

test = np.array([1, -1, 1, 1])

for _ in range(5):
    test = np.sign(W @ test)

print("Recovered Pattern:", test)
