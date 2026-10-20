# Exercise 3 - Multi-class (3 classes, 4 features) with the SAME MLP class as Exercise 2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mlp import MLP, one_hot

rng = np.random.default_rng(42)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- A. Data ----------
X, y = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                           n_repeated=0, n_classes=3, n_clusters_per_class=2,
                           class_sep=1.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- C. Training (only the output size changes: 3 units -> softmax) ----------
sizes, lr, epochs = [4, 16, 3], 0.3, 5000
mlp = MLP(sizes, rng, lr=lr)
losses = mlp.fit(X_tr, one_hot(y_tr, 3), epochs)
y_pred = mlp.predict(X_te)
acc_mlp = np.mean(y_pred == y_te)
print(f"MLP {sizes}, lr={lr}, epochs={epochs}: final train loss {losses[-1]:.4f}, "
      f"test accuracy {acc_mlp:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(losses, label="train loss (cross-entropy)")
plt.title("Figure 4 - Training loss (3-class MLP)")
plt.xlabel("epoch"); plt.ylabel("categorical cross-entropy"); plt.legend()
plt.savefig(f"{FIG}/fig4.png", dpi=120); plt.close()

cm = confusion_matrix(y_te, y_pred)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1", "Class 2"]).plot(cmap="Blues")
plt.title("Figure 5 - Confusion matrix (test set)")
plt.xlabel("predicted class"); plt.ylabel("true class")
plt.savefig(f"{FIG}/fig5.png", dpi=120); plt.close()

# ---------- D. Linear baseline ----------
acc_lin = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
print(f"Logistic regression test accuracy: {acc_lin:.4f}")
