# VAE on MNIST (shared by Exercises 1 and 2).
# Hand-written parts: reparameterization, BCE reconstruction, closed-form KL, sampling.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")          # MNIST is downloaded here
FIG = os.path.join(HERE, "..", "figures")


def load_mnist():
    """Train (50k) / validation (10k) / test (10k). Images are 784-vectors in [0, 1]."""
    tf = transforms.Compose([transforms.ToTensor(),                  # pixels in [0, 1]
                             transforms.Lambda(lambda x: x.view(-1))])  # 28x28 -> 784
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [50000, 10000],
                              generator=torch.Generator().manual_seed(42))
    return train, val, test


class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        # encoder 784 -> 512 -> 256 (ReLU), then two heads 256 -> d
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        # decoder d -> 256 -> 512 -> 784, sigmoid output (pixel probabilities)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                 nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, eps ~ N(0, I).  sigma = exp(logvar / 2)
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(x_hat, x, mu, logvar):
    """Both terms in nats per image (summed over pixels / latent dims, averaged over the batch)."""
    n = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / n
    kl = 0.5 * torch.sum(mu ** 2 + logvar.exp() - 1 - logvar) / n
    return recon, kl


@torch.no_grad()
def evaluate(model, dataset):
    """Average reconstruction and KL (nats/image) over a whole dataset."""
    model.eval()
    recon_sum, kl_sum, n = 0.0, 0.0, 0
    for x, _ in DataLoader(dataset, batch_size=1000):
        x_hat, mu, logvar = model(x)
        recon, kl = loss_terms(x_hat, x, mu, logvar)
        recon_sum += recon.item() * len(x)
        kl_sum += kl.item() * len(x)
        n += len(x)
    return recon_sum / n, kl_sum / n


def train_vae(d, train, val, epochs=20, beta=1.0, lr=1e-3, batch_size=128):
    torch.manual_seed(42)                       # seed before building each model
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    hist = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}

    for epoch in range(epochs):
        model.train()
        recon_sum, kl_sum, n = 0.0, 0.0, 0
        for x, _ in loader:
            x_hat, mu, logvar = model(x)
            recon, kl = loss_terms(x_hat, x, mu, logvar)
            loss = recon + beta * kl            # negative ELBO (beta = 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_sum += recon.item() * len(x)
            kl_sum += kl.item() * len(x)
            n += len(x)
        val_recon, val_kl = evaluate(model, val)
        hist["train_recon"].append(recon_sum / n)
        hist["train_kl"].append(kl_sum / n)
        hist["val_recon"].append(val_recon)
        hist["val_kl"].append(val_kl)
        print(f"d={d} epoch {epoch + 1:2d}: train recon {recon_sum / n:.2f}  KL {kl_sum / n:.2f} | "
              f"val recon {val_recon:.2f}  KL {val_kl:.2f}")
    return model, hist
