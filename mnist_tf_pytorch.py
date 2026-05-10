import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# PART 1: MNIST Handwritten Character Detection using TensorFlow / Keras
# =============================================================================

print("=" * 55)
print("  PART 1: TensorFlow / Keras — MNIST")
print("=" * 55)

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Build model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss    : {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")


# =============================================================================
# PART 2: MNIST Handwritten Character Detection using PyTorch
# =============================================================================

print()
print("=" * 55)
print("  PART 2: PyTorch — MNIST")
print("=" * 55)

# Transform
transform = transforms.ToTensor()

# Load dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Model
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model_pt = NN()

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)

# Training
for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        outputs = model_pt(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/5 completed — Loss: {running_loss:.4f}")

print("Training Done")

# Testing
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model_pt(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
