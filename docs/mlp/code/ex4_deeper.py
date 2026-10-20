# Exercise 4 - Does a second hidden layer help?
# Same data, learning rate and epochs as Exercise 3. Only the depth changes.
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlp import MLP, one_hot

rng = np.random.default_rng(42)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

X, y = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                           n_repeated=0, n_classes=3, n_clusters_per_class=2,
                           class_sep=1.0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr, epochs = 0.3, 5000
models = {"Ex. 3 (1 hidden)": [4, 16, 3],
          "Ex. 4 (2 hidden)": [4, 16, 16, 3]}
n_runs = 3   # 3 different random initialisations (the same rng keeps advancing)

results = {name: {"loss": [], "final": [], "acc": []} for name in models}
for run in range(n_runs):
    for name, sizes in models.items():
        mlp = MLP(sizes, rng, lr=lr)
        losses = mlp.fit(X_tr, one_hot(y_tr, 3), epochs)
        acc = np.mean(mlp.predict(X_te) == y_te)
        results[name]["loss"].append(losses)
        results[name]["final"].append(losses[-1])
        results[name]["acc"].append(acc)
        results[name]["params"] = mlp.n_params()
        print(f"run {run}  {name} {sizes}: final train loss {losses[-1]:.4f}, test acc {acc:.4f}")

print("\n| Model | Hidden layers | Parameters | Final train loss (mean) | Test accuracy (mean ± std) |")
print("|---|---|---|---|---|")
for name, sizes in models.items():
    r = results[name]
    print(f"| {name} {sizes} | {len(sizes) - 2} | {r['params']} | {np.mean(r['final']):.4f} | "
          f"{np.mean(r['acc']):.4f} ± {np.std(r['acc']):.4f} |")
diff = np.mean(results["Ex. 4 (2 hidden)"]["acc"]) - np.mean(results["Ex. 3 (1 hidden)"]["acc"])
print(f"\nAccuracy difference (Ex.4 - Ex.3), mean over {n_runs} runs: {diff:+.4f}")

plt.figure(figsize=(7, 4))
for (name, r), color in zip(results.items(), ["C0", "C1"]):
    mean_loss = np.mean(r["loss"], axis=0)
    plt.plot(mean_loss, color=color, label=f"{name} (mean of {n_runs} runs)")
plt.title("Figure 6 - Training loss: 1 vs 2 hidden layers")
plt.xlabel("epoch"); plt.ylabel("categorical cross-entropy"); plt.legend()
plt.savefig(f"{FIG}/fig6.png", dpi=120); plt.close()
