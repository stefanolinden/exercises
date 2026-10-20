# Exercise 2 - A latent space you can see (d = 2)
# Run ex1_vae_d16.py first: Figure 6 uses the saved d = 16 model.
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
from vae import VAE, load_mnist, train_vae, evaluate, FIG, HERE

train, val, test = load_mnist()

# ---------- A. Same model and training, only d = 2 ----------
model, hist = train_vae(d=2, train=train, val=val)
test_recon, test_kl = evaluate(model, test)
print(f"TEST d=2: recon {test_recon:.2f}, KL {test_kl:.2f}, negative ELBO {test_recon + test_kl:.2f}")
model.eval()

# ---------- B. Encoded test set (mu of every test image) ----------
with torch.no_grad():
    x_all, y_all = next(iter(DataLoader(test, batch_size=len(test))))
    mu_all, _ = model.encode(x_all)
mu_all, y_all = mu_all.numpy(), y_all.numpy()

plt.figure(figsize=(8, 7))
for k in range(10):
    plt.scatter(mu_all[y_all == k, 0], mu_all[y_all == k, 1], s=2, color=plt.cm.tab10(k), label=str(k))
plt.title("Figure 4 - Test set encoded by the d = 2 VAE (mu)")
plt.xlabel("z1"); plt.ylabel("z2")
plt.legend(title="digit", markerscale=6)
plt.savefig(f"{FIG}/fig4.png", dpi=120); plt.close()

# ---------- C. Decoded plane on a 15x15 grid spaced by the prior's quantiles ----------
q = np.linspace(0.05, 0.95, 15)
z_vals = norm.ppf(q)                      # inverse CDF of N(0, 1)
canvas = np.zeros((15 * 28, 15 * 28))
with torch.no_grad():
    for r, z2 in enumerate(z_vals[::-1]):          # top row = largest z2 (like a plot)
        for c, z1 in enumerate(z_vals):            # left column = smallest z1
            img = model.decode(torch.tensor([[z1, z2]], dtype=torch.float32))
            canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img.view(28, 28).numpy()

ticks = np.arange(15) * 28 + 14
plt.figure(figsize=(9, 9))
plt.imshow(canvas, cmap="gray")
plt.xticks(ticks, [f"{v:.1f}" for v in z_vals], fontsize=7)
plt.yticks(ticks, [f"{v:.1f}" for v in z_vals[::-1]], fontsize=7)
plt.title("Figure 5 - Decoded 15x15 grid, z = Phi^-1(q), q in [0.05, 0.95]")
plt.xlabel("z1"); plt.ylabel("z2")
plt.savefig(f"{FIG}/fig5.png", dpi=120); plt.close()

# ---------- D. Interpolation: pixel space vs latent space (d = 16 model) ----------
ia, ib = 3, 2                      # test index 3 is a "0", test index 2 is a "1"
print(f"Interpolating test index {ia} (digit {test[ia][1]}) -> index {ib} (digit {test[ib][1]})")
xa, xb = test[ia][0], test[ib][0]

model16 = VAE(16)
model16.load_state_dict(torch.load(os.path.join(HERE, "vae_d16.pt")))
model16.eval()

ts = np.linspace(0, 1, 10)
with torch.no_grad():
    mu_a, _ = model16.encode(xa.unsqueeze(0))
    mu_b, _ = model16.encode(xb.unsqueeze(0))
    pixel_row = [(1 - t) * xa + t * xb for t in ts]
    latent_row = [model16.decode((1 - t) * mu_a + t * mu_b)[0] for t in ts]

fig, ax = plt.subplots(2, 10, figsize=(12, 3))
for j, t in enumerate(ts):
    ax[0, j].imshow(pixel_row[j].view(28, 28), cmap="gray")
    ax[1, j].imshow(latent_row[j].view(28, 28), cmap="gray")
    ax[0, j].set_title(f"t={t:.2f}", fontsize=8)
    ax[0, j].axis("off"); ax[1, j].axis("off")
ax[0, 0].text(-14, 14, "pixels", rotation=90, va="center")
ax[1, 0].text(-14, 14, "latent", rotation=90, va="center")
fig.suptitle(f"Figure 6 - Interpolation test #{ia} -> #{ib}: pixel space (top) vs latent space, d = 16 (bottom)")
fig.savefig(f"{FIG}/fig6.png", dpi=120); plt.close(fig)
