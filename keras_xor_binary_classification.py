import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


# =============================================================================
# PART 1: XOR Classification using Neural Network
# =============================================================================

print("=" * 50)
print("  PART 1: XOR Classification (Neural Network)")
print("=" * 50)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR

model = models.Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=0)

loss, acc = model.evaluate(X, y, verbose=0)
print("Loss    :", round(loss, 4))
print("Accuracy:", round(acc, 4))


# =============================================================================
# PART 2: Binary Classification with Confusion Matrix
# =============================================================================

print()
print("=" * 50)
print("  PART 2: Binary Classification")
print("=" * 50)

X2 = np.array([[2], [4], [6], [8], [10]])
y2 = np.array([0, 0, 0, 1, 1])

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
])

model2.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

model2.fit(X2, y2, epochs=100, verbose=0)

y_pred_prob = model2.predict(X2, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc2 = accuracy_score(y2, y_pred)
cm = confusion_matrix(y2, y_pred)

print("Accuracy        :", round(acc2, 4))
print("Confusion Matrix:")
print(cm)
