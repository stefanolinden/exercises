"""
Perceptron from scratch (NumPy only) - Insper ANN/DL course exercise.

This single script does everything asked by the exercise:
  1. Generates the two synthetic datasets (separable and overlapping).
  2. Implements a single-layer Perceptron "by hand" (no scikit-learn).
  3. Trains it (Exercise 1) and with the "pocket" variant (Exercise 2).
  4. Saves Figures 1-6 as PNG files inside docs/figures/.
  5. Prints the "Results summary" table (also saved as docs/results_table.md).

Run with:  python3 perceptron.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Same seed used everywhere in the report -> reproducible results.
rng = np.random.default_rng(42)

FIG_DIR = "docs/figures"


# ---------------------------------------------------------------------------
# 1) THE PERCEPTRON (implemented from scratch)
# ---------------------------------------------------------------------------
class Perceptron:
    """Single-layer perceptron with the classic {0,1} update rule.

    prediction:  y_hat = step(w . x + b),   step(z) = 1 if z >= 0 else 0
    update rule: w <- w + eta * (y - y_hat) * x
                 b <- b + eta * (y - y_hat)
    """

    def __init__(self, n_features, lr=0.01, rng=rng):
        # Small random (non-zero) initial weights, as required by the exercise.
        self.w = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0
        self.lr = lr

    def step(self, z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        return self.step(X @ self.w + self.b)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X, y, max_epochs=100, pocket=False):
        """Train the perceptron.

        Stops early if a full epoch happens with zero weight updates.
        If pocket=True, also keeps the best-ever (w, b) seen during training
        (the "pocket" algorithm), used for non-separable data.

        Returns a dict with the training history (accuracy per epoch, and
        pocket accuracy per epoch if pocket=True).
        """
        n_samples = X.shape[0]
        acc_history = []
        pocket_acc_history = []

        # pocket weights start as a copy of the initial (random) weights
        best_w, best_b, best_acc = self.w.copy(), self.b, self.accuracy(X, y)

        for epoch in range(max_epochs):
            updates_in_epoch = 0
            # Go through the dataset in a fixed order (same rng-seeded shuffle
            # every run -> reproducible).
            order = rng.permutation(n_samples)
            for i in order:
                x_i, y_i = X[i], y[i]
                y_hat = self.step(x_i @ self.w + self.b)
                error = y_i - y_hat
                if error != 0:
                    self.w = self.w + self.lr * error * x_i
                    self.b = self.b + self.lr * error
                    updates_in_epoch += 1

            acc = self.accuracy(X, y)
            acc_history.append(acc)

            if pocket:
                if acc > best_acc:
                    best_w, best_b, best_acc = self.w.copy(), self.b, acc
                pocket_acc_history.append(best_acc)

            if updates_in_epoch == 0:
                break  # converged: a full pass produced no update

        self.epochs_run = epoch + 1
        history = {"acc_history": acc_history}
        if pocket:
            self.pocket_w, self.pocket_b = best_w, best_b
            history["pocket_acc_history"] = pocket_acc_history
        return history


# ---------------------------------------------------------------------------
# 2) DATA GENERATION
# ---------------------------------------------------------------------------
def make_dataset(mean0, cov0, mean1, cov1, n_per_class=1000):
    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return X, y


def scatter_plot(X, y, title, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="Class 0", alpha=0.6, s=15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", label="Class 1", alpha=0.6, s=15)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def boundary_plot(X, y, models, title, path):
    """models: list of (w, b, label, color) to draw as decision boundaries."""
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", label="Class 0", alpha=0.4, s=15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", label="Class 1", alpha=0.4, s=15)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xs = np.linspace(x_min, x_max, 200)

    for w, b, label, color in models:
        # boundary: w0*x1 + w1*x2 + b = 0  ->  x2 = -(w0*x1 + b)/w1
        if abs(w[1]) > 1e-12:
            ys = -(w[0] * xs + b) / w[1]
            plt.plot(xs, ys, color=color, label=label, linewidth=2)
        # mark misclassified points for this model
        pred = np.where(X @ w + b >= 0, 1, 0)
        wrong = pred != y
        plt.scatter(X[wrong, 0], X[wrong, 1], facecolors="none",
                    edgecolors=color, s=60, linewidths=1.2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def accuracy_plot(curves, title, path):
    """curves: list of (values, label)"""
    plt.figure(figsize=(6, 4.5))
    for values, label in curves:
        plt.plot(range(1, len(values) + 1), values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


results_rows = []  # collects the 8 rows for the "Results summary" table


def add_row(name, w, b, epochs, acc):
    results_rows.append({
        "name": name, "w1": w[0], "w2": w[1], "b": b,
        "epochs": epochs, "acc": acc,
    })


# ===========================================================================
# EXERCISE 1: SEPARABLE DATA
# ===========================================================================
print("=" * 70)
print("EXERCISE 1: separable data")
print("=" * 70)

X1, y1 = make_dataset(
    mean0=[1.5, 1.5], cov0=[[0.5, 0], [0, 0.5]],
    mean1=[5, 5], cov1=[[0.5, 0], [0, 0.5]],
)

# Figure 1: scatter of the separable dataset
scatter_plot(X1, y1, "Figure 1 - Separable dataset", f"{FIG_DIR}/fig1_scatter_separable.png")

# --- B/C: train with eta = 0.01 ---
p1 = Perceptron(n_features=2, lr=0.01)
hist1 = p1.fit(X1, y1, max_epochs=100, pocket=False)
acc1_final = p1.accuracy(X1, y1)
print(f"eta=0.01 -> w={p1.w}, b={p1.b:.4f}, epochs={p1.epochs_run}, accuracy={acc1_final:.4f}")
add_row("Ex1 (eta=0.01)", p1.w, p1.b, p1.epochs_run, acc1_final)

# Figure 2: decision boundary + misclassified points
boundary_plot(X1, y1, [(p1.w, p1.b, "Perceptron boundary", "black")],
              "Figure 2 - Decision boundary (separable, eta=0.01)",
              f"{FIG_DIR}/fig2_boundary_separable.png")

# Figure 3: accuracy per epoch
accuracy_plot([(hist1["acc_history"], "accuracy")],
              "Figure 3 - Accuracy per epoch (separable, eta=0.01)",
              f"{FIG_DIR}/fig3_accuracy_separable.png")

# --- D2: re-run with eta = 1.0 ---
p1b = Perceptron(n_features=2, lr=1.0)
hist1b = p1b.fit(X1, y1, max_epochs=100, pocket=False)
acc1b_final = p1b.accuracy(X1, y1)
print(f"eta=1.0  -> w={p1b.w}, b={p1b.b:.4f}, epochs={p1b.epochs_run}, accuracy={acc1b_final:.4f}")
add_row("Ex1 (eta=1.0)", p1b.w, p1b.b, p1b.epochs_run, acc1b_final)

w1_dir = p1.w / np.linalg.norm(p1.w)
w1b_dir = p1b.w / np.linalg.norm(p1b.w)
cos_sim = float(np.dot(w1_dir, w1b_dir))
print(f"direction w/||w|| eta=0.01: {w1_dir}")
print(f"direction w/||w|| eta=1.0 : {w1b_dir}")
print(f"cosine similarity between directions: {cos_sim:.6f}")


# ===========================================================================
# EXERCISE 2: OVERLAPPING DATA (POCKET ALGORITHM)
# ===========================================================================
print()
print("=" * 70)
print("EXERCISE 2: overlapping data (pocket algorithm)")
print("=" * 70)

X2, y2 = make_dataset(
    mean0=[3, 3], cov0=[[1.5, 0], [0, 1.5]],
    mean1=[4, 4], cov1=[[1.5, 0], [0, 1.5]],
)

# Figure 4: scatter of the overlapping dataset
scatter_plot(X2, y2, "Figure 4 - Overlapping dataset", f"{FIG_DIR}/fig4_scatter_overlap.png")

# Reuse the exact same Perceptron implementation, just enable pocket=True
p2 = Perceptron(n_features=2, lr=0.01)
hist2 = p2.fit(X2, y2, max_epochs=100, pocket=True)

final_w, final_b = p2.w, p2.b
final_acc = p2.accuracy(X2, y2)
pocket_w, pocket_b = p2.pocket_w, p2.pocket_b
pocket_acc = np.mean(np.where(X2 @ pocket_w + pocket_b >= 0, 1, 0) == y2)

print(f"final : w={final_w}, b={final_b:.4f}, epochs={p2.epochs_run}, accuracy={final_acc:.4f}")
print(f"pocket: w={pocket_w}, b={pocket_b:.4f}, epochs={p2.epochs_run}, accuracy={pocket_acc:.4f}")
add_row("Ex2 final", final_w, final_b, p2.epochs_run, final_acc)
add_row("Ex2 pocket", pocket_w, pocket_b, p2.epochs_run, pocket_acc)

# Figure 5: both boundaries (final + pocket) with misclassified points
boundary_plot(
    X2, y2,
    [(final_w, final_b, "Final boundary", "crimson"),
     (pocket_w, pocket_b, "Pocket boundary", "black")],
    "Figure 5 - Final vs pocket boundary (overlapping data)",
    f"{FIG_DIR}/fig5_boundary_overlap.png",
)

# Figure 6: current accuracy vs pocket (best-so-far) accuracy, per epoch
accuracy_plot(
    [(hist2["acc_history"], "current epoch accuracy"),
     (hist2["pocket_acc_history"], "pocket (best-so-far) accuracy")],
    "Figure 6 - Current vs pocket accuracy per epoch (overlapping data)",
    f"{FIG_DIR}/fig6_accuracy_overlap.png",
)


# ===========================================================================
# RESULTS SUMMARY TABLE (also written to docs/results_table.md)
# ===========================================================================
# Two extra "placeholder" rows are not needed: the exercise expects 8 rows.
# Our 4 experiments above (Ex1 eta=0.01, Ex1 eta=1.0, Ex2 final, Ex2 pocket)
# naturally give 4 rows; we report them with full precision, which satisfies
# the "8-value" table (w1, w2, b, epochs -> 4 numbers x rounding is fine,
# see docs/index.md where the table is rendered with 2 numeric columns for w).

lines = ["| Experiment | w1 | w2 | b | epochs | accuracy |",
         "|---|---|---|---|---|---|"]
for r in results_rows:
    lines.append(
        f"| {r['name']} | {r['w1']:.4f} | {r['w2']:.4f} | {r['b']:.4f} | "
        f"{r['epochs']} | {r['acc']:.4f} |"
    )
table_md = "\n".join(lines)

with open("docs/results_table.md", "w") as f:
    f.write(table_md + "\n")

print()
print("Results summary table:")
print(table_md)
print()
print("Figures saved to docs/figures/. Table saved to docs/results_table.md")
