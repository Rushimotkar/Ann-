import numpy as np

X = np.array([[48], [49], [50], [51], [52], [53], [54], [55], [56], [57]])

y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

weights = np.zeros(1)
bias = 0
learning_rate = 0.01

def step_function(x):
    return 1 if x >= 0 else 0

epochs = 10

for _ in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = step_function(linear_output)

        # Update weights and bias
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Training complete!")
print("Weights:", weights)
print("Bias:", bias)

test_input = int(input("Enter ASCII value of a digit (48–57): "))
prediction = step_function(np.dot([test_input], weights) + bias)

if prediction == 0:
    print("The number is EVEN")
else:
    print("The number is ODD")
