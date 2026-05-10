import numpy as np

X = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])
Y = np.array([
    [1, 1],
    [-1, -1]
])

W = np.zeros((X.shape[1], Y.shape[1]))

for i in range(len(X)):
    W += np.outer(X[i], Y[i])
print("Weight Matrix W:\n", W)

def sign(x):
    return np.where(x >= 0, 1, -1)

def bam_recall(input_vector, W, steps=5):
    x = input_vector.copy()
    for _ in range(steps):
        y = sign(np.dot(x, W))       # Forward: X → Y
        x = sign(np.dot(y, W.T))     # Backward: Y → X
    return x, y

test_input = np.array([1, -1, 1])
final_x, final_y = bam_recall(test_input, W)

print("\nInput X:", test_input)
print("Recalled X:", final_x)
print("Recalled Y:", final_y)
