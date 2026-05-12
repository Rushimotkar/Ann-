# Naive Bayes Classification on Iris Dataset
# Computes: Confusion Matrix, TP, FP, TN, FN,
#           Accuracy, Error Rate, Precision, Recall

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("Iris.csv")
print("Dataset Head:")
print(df.head())

# ── 2. Features & Target ─────────────────────────────────────
X = df.drop('Species', axis=1)
y = df['Species']

# ── 3. Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train Naive Bayes Model ───────────────────────────────
model = GaussianNB()
model.fit(X_train, y_train)

# ── 5. Predictions ───────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 6. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

classes = model.classes_
print(f"\nClasses: {classes}")

# ── 7. TP, FP, TN, FN (per class) ───────────────────────────
print("\n--- Per-Class Metrics ---")
for i, cls in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) != 0 else 0

    print(f"\nClass: {cls}")
    print(f"  TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Error Rate : {error_rate:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")

# ── 8. Overall Accuracy ──────────────────────────────────────
overall_accuracy = (y_test == y_pred).sum() / len(y_test)
print(f"\nOverall Accuracy : {overall_accuracy:.4f}")
print(f"Overall Error Rate: {1 - overall_accuracy:.4f}")

# ── 9. Classification Report ─────────────────────────────────
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
