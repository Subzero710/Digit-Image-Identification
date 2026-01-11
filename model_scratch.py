import numpy as np
from my_descent_gradient import GradientDescent


class ModelScratch:
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        self.theta = None

    def _softmax(self, Z):
        # fonction de prédiction(g_Theta) de notre model logistique "From Scratch"
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        g_Theta =exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return g_Theta

    def _one_hot(self, y, C):
        return np.eye(C)[y.astype(int)]

    def fit(self, X, y):
        N = X.shape[0]
        C = 10
        #on ajoute 1 au debut pour le biais
        X_bias = np.hstack((np.ones((N, 1)), X))
        Y_encoded = self._one_hot(y, C)

        initial_theta = np.zeros((C, X_bias.shape[1]))

        # formule du gradient moyen sous forme matricielle
        def grad_func(current_theta):
            Pi = self._softmax(X_bias @ current_theta.T)
            return (1 / N) * ((Pi - Y_encoded).T @ X_bias)

        optimizer = GradientDescent(gradient=grad_func,
                                    learning_rate=self.lr,
                                    max_iterations=self.iterations)

        self.theta = optimizer.descent(initial_theta)

    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        # règle de décision f(x)=argmax(Theta*x)
        probas = self._softmax(X_bias @ self.theta.T)
        return np.argmax(probas, axis=1)