# Exercise 2 - Binary classification (2 blobs of class 1 around class 0)
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlp import MLP

rng = np.random.default_rng(42)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

# ---------- A. Data ----------
X, c = make_blobs(n_samples=[500, 250, 250], centers=[[0, 0], [3, 3], [-3, -3]],
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)        # cluster 0 -> class 0, clusters 1 and 2 -> class 1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

plt.figure(figsize=(6, 5))
for k in [0, 1]:
    plt.scatter(X[y == k, 0], X[y == k, 1], s=10, label=f"Class {k}")
plt.title("Figure 1 - Binary dataset (1000 points)")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.savefig(f"{FIG}/fig1.png", dpi=120); plt.close()

# ---------- B. Linear baseline ----------
logreg = LogisticRegression().fit(X_tr, y_tr)
acc_lin = logreg.score(X_te, y_te)
print(f"Logistic regression test accuracy: {acc_lin:.4f}")

# ---------- C. MLP from scratch ----------
sizes, lr, epochs = [2, 8, 1], 0.5, 2000
mlp = MLP(sizes, rng, lr=lr)
losses = mlp.fit(X_tr, y_tr.reshape(-1, 1), epochs)
acc_mlp = np.mean(mlp.predict(X_te) == y_te)
print(f"MLP {sizes}, lr={lr}, epochs={epochs}: final train loss {losses[-1]:.4f}, "
      f"test accuracy {acc_mlp:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(losses, label="train loss (BCE)")
plt.title("Figure 2 - Training loss (binary MLP)")
plt.xlabel("epoch"); plt.ylabel("binary cross-entropy"); plt.legend()
plt.savefig(f"{FIG}/fig2.png", dpi=120); plt.close()

# decision boundary: predict on a grid and shade the regions
xx, yy = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-7, 7, 300))
grid_pred = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, grid_pred, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=["C0", "C1"])
for k in [0, 1]:
    plt.scatter(X_te[y_te == k, 0], X_te[y_te == k, 1], s=12, label=f"Class {k} (test)")
plt.title("Figure 3 - MLP decision boundary (test points)")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
plt.savefig(f"{FIG}/fig3.png", dpi=120); plt.close()
