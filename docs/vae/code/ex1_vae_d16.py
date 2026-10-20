# Exercise 1 - VAE on MNIST with d = 16
import os
import torch
import matplotlib.pyplot as plt
from vae import load_mnist, train_vae, evaluate, FIG, HERE

# ---------- A. Data ----------
train, val, test = load_mnist()
print(f"train {len(train)}, val {len(val)}, test {len(test)}")

# ---------- B + C. Model and training ----------
model, hist = train_vae(d=16, train=train, val=val)
torch.save(model.state_dict(), os.path.join(HERE, "vae_d16.pt"))   # reused in Exercise 2D

# test set used only here, for the final numbers
test_recon, test_kl = evaluate(model, test)
print(f"TEST d=16: recon {test_recon:.2f}, KL {test_kl:.2f}, negative ELBO {test_recon + test_kl:.2f}")
print(f"Validation KL: first epoch {hist['val_kl'][0]:.2f}, last epoch {hist['val_kl'][-1]:.2f}")

# Figure 1: the two loss terms per epoch
epochs = range(1, len(hist["train_recon"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(epochs, hist["train_recon"], label="train")
ax[0].plot(epochs, hist["val_recon"], label="validation")
ax[0].set_title("Reconstruction (BCE) vs epoch")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("nats per image"); ax[0].legend()
ax[1].plot(epochs, hist["train_kl"], label="train")
ax[1].plot(epochs, hist["val_kl"], label="validation")
ax[1].set_title("KL vs epoch")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("nats per image"); ax[1].legend()
fig.suptitle("Figure 1 - VAE (d = 16) loss terms")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1.png", dpi=120); plt.close(fig)

# ---------- D. Reconstructions and samples ----------
model.eval()
with torch.no_grad():
    # first test image of each digit 0..9
    labels = test.targets
    idx = [int((labels == k).nonzero()[0]) for k in range(10)]
    print("First test index of each digit 0-9:", idx)
    x = torch.stack([test[i][0] for i in idx])
    mu, _ = model.encode(x)
    x_rec = model.decode(mu)                  # decode the mean, not a sample

    fig, ax = plt.subplots(2, 10, figsize=(12, 2.8))
    for k in range(10):
        ax[0, k].imshow(x[k].view(28, 28), cmap="gray")
        ax[1, k].imshow(x_rec[k].view(28, 28), cmap="gray")
        ax[0, k].set_title(str(k))
        ax[0, k].axis("off"); ax[1, k].axis("off")
    ax[0, 0].text(-12, 14, "original", rotation=90, va="center")
    ax[1, 0].text(-12, 14, "reconstr.", rotation=90, va="center")
    fig.suptitle("Figure 2 - Test images (top) and reconstructions from mu (bottom)")
    fig.savefig(f"{FIG}/fig2.png", dpi=120); plt.close(fig)

    # 64 samples from the prior z ~ N(0, I)
    torch.manual_seed(0)
    z = torch.randn(64, 16)
    samples = model.decode(z)
    fig, ax = plt.subplots(8, 8, figsize=(7, 7.4))
    for i in range(64):
        ax[i // 8, i % 8].imshow(samples[i].view(28, 28), cmap="gray")
        ax[i // 8, i % 8].axis("off")
    fig.suptitle("Figure 3 - 64 samples decoded from z ~ N(0, I)")
    fig.savefig(f"{FIG}/fig3.png", dpi=120); plt.close(fig)
