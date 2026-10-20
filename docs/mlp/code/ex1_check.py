# Exercise 1 is done by hand in the report. This script only CHECKS the hand numbers.
import numpy as np

x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# A. forward pass
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
L = (y - y_hat) ** 2
print("z1 =", z1.round(4), " h1 =", h1.round(4))
print("u2 =", round(u2, 4), " y_hat =", round(y_hat, 4), " L =", round(L, 4))

# B. backward pass
dL_dyhat = 2 * (y_hat - y)
dL_du2 = dL_dyhat * (1 - y_hat ** 2)        # tanh'(u) = 1 - tanh(u)^2
dW2 = dL_du2 * h1
db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1 ** 2)
dW1 = np.outer(dL_dz1, x)
db1 = dL_dz1
print("dL/dyhat =", round(dL_dyhat, 4), " dL/du2 =", round(dL_du2, 4))
print("dW2 =", dW2.round(4), " db2 =", round(db2, 4))
print("dL/dh1 =", dL_dh1.round(4), " dL/dz1 =", dL_dz1.round(4))
print("dW1 =\n", dW1.round(4), "\ndb1 =", db1.round(4))

# C. update
W2n, b2n = W2 - eta * dW2, b2 - eta * db2
W1n, b1n = W1 - eta * dW1, b1 - eta * db1
print("new W2 =", W2n.round(4), " new b2 =", round(b2n, 4))
print("new W1 =\n", W1n.round(4), "\nnew b1 =", b1n.round(4))

# second forward pass with the new parameters
y_hat_new = np.tanh(W2n @ np.tanh(W1n @ x + b1n) + b2n)
print("new y_hat =", round(y_hat_new, 4), " new L =", round((y - y_hat_new) ** 2, 4))
