# MLP written from scratch with numpy (used by Exercises 2, 3 and 4).
#
# - Hidden layers: tanh
# - Output layer:  sigmoid + binary cross-entropy  (1 output unit)
#                  softmax + categorical cross-entropy (K output units)
#
# Both output/loss pairs have the SAME gradient w.r.t. the output pre-activation:
#     dL/du = (y_hat - y) / N
# so forward(), backward() and update() are exactly the same code for
# binary and multi-class problems. Only the final activation changes.
import numpy as np


class MLP:
    def __init__(self, layer_sizes, rng, lr=0.1):
        # layer_sizes example: [2, 8, 1] = 2 inputs, 8 hidden, 1 output
        self.lr = lr
        self.W = []
        self.b = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # small random weights (Xavier-like scale), zero biases
            self.W.append(rng.normal(0, 1 / np.sqrt(n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def output_activation(self, u):
        if u.shape[1] == 1:                        # binary -> sigmoid
            return 1 / (1 + np.exp(-u))
        u = u - u.max(axis=1, keepdims=True)       # multi-class -> softmax (stable)
        e = np.exp(u)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        # keep every layer output (needed by backprop)
        self.h = [X]
        for i in range(len(self.W)):
            u = self.h[-1] @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                self.h.append(np.tanh(u))                    # hidden layer
            else:
                self.h.append(self.output_activation(u))     # output layer
        return self.h[-1]

    def loss(self, y_hat, Y):
        eps = 1e-12
        if Y.shape[1] == 1:   # binary cross-entropy
            return -np.mean(Y * np.log(y_hat + eps) + (1 - Y) * np.log(1 - y_hat + eps))
        return -np.mean(np.sum(Y * np.log(y_hat + eps), axis=1))  # categorical CE

    def backward(self, Y):
        N = Y.shape[0]
        delta = (self.h[-1] - Y) / N          # dL/du of the output layer
        self.dW = [None] * len(self.W)
        self.db = [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            self.dW[i] = self.h[i].T @ delta
            self.db[i] = delta.sum(axis=0)
            if i > 0:
                # go back one layer: through the weights, then through tanh
                delta = (delta @ self.W[i].T) * (1 - self.h[i] ** 2)

    def update(self):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.dW[i]
            self.b[i] -= self.lr * self.db[i]

    def fit(self, X, Y, epochs):
        losses = []
        for _ in range(epochs):
            y_hat = self.forward(X)
            losses.append(self.loss(y_hat, Y))
            self.backward(Y)
            self.update()
        return losses

    def predict(self, X):
        y_hat = self.forward(X)
        if y_hat.shape[1] == 1:
            return (y_hat[:, 0] > 0.5).astype(int)
        return y_hat.argmax(axis=1)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]
