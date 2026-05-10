import numpy as np

zero = np.array([
    1, 1, 1,
    1, 0, 1,
    1, 0, 1,
    1, 0, 1,
    1, 1, 1
])
one = np.array([
    0, 1, 0,
    1, 1, 0,
    0, 1, 0,
    0, 1, 0,
    1, 1, 1
])
two = np.array([
    1, 1, 1,
    0, 0, 1,
    1, 1, 1,
    1, 0, 0,
    1, 1, 1
])
three = np.array([
    1, 1, 1,
    0, 0, 1,
    1, 1, 1,
    0, 0, 1,
    1, 1, 1
])

X = np.array([zero, one, two, three])
Y = np.array([
    [1, 0, 0, 0],  # 0
    [0, 1, 0, 0],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 0, 1]   # 3
])

W = np.dot(X.T, Y)
print("Weight Matrix:\n", W)

def predict(x):
    net = np.dot(x, W)
    return np.argmax(net)

# Test on all training patterns
print("\nTesting on training patterns:")
for i, (pattern, label) in enumerate(zip(X, [0, 1, 2, 3])):
    result = predict(pattern)
    print(f"Input: {label} → Predicted: {result} {'✓' if result == label else '✗'}")
