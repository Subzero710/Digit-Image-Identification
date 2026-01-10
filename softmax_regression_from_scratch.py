import numpy as np

def add_bias(X):
    return np.c_[np.ones(len(X)), X]

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

class SoftmaxRegressionGD:
    def __init__(self, lr=0.1, epochs=5000, tol=1e-6, l2=0.0, seed=0, print_every=200, verbose=True):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.l2 = l2
        self.seed = seed
        self.print_every = print_every
        self.verbose = verbose
        self.theta = None
        self.losses = []
        self.grad_norms = []

    def fit(self, X, y, C=None):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        N, d = X.shape
        C = int(C if C is not None else y.max() + 1)
        rng = np.random.default_rng(self.seed)
        self.theta = rng.normal(0.0, 0.01, size=(d, C))
        Y = np.eye(C)[y]
        eps = 1e-12

        for e in range(1, self.epochs + 1):
            P = softmax(X @ self.theta)
            loss = -np.sum(Y * np.log(P + eps)) / N
            if self.l2:
                loss = loss + 0.5 * self.l2 * np.sum(self.theta[1:] ** 2)

            G = (X.T @ (P - Y)) / N
            if self.l2:
                G[1:] = G[1:] + self.l2 * self.theta[1:]

            gnorm = float(np.linalg.norm(G))
            self.losses.append(float(loss))
            self.grad_norms.append(gnorm)

            if self.verbose and (e == 1 or e % self.print_every == 0):
                print(f"epoch={e:5d} loss={loss:.6f} grad_norm={gnorm:.3e}")

            if gnorm <= self.tol:
                break

            self.theta = self.theta - self.lr * G
        return self

    def predict_proba(self, X):
        return softmax(np.asarray(X, float) @ self.theta)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)