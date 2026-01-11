import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from model_scratch import ModelScratch
from analysis import plot_coefficients, plot_confusion
from analyze_digits_results import report, show_examples

# Découpage du set de données : 80% entrainement / 20% evaluation
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, digits.images, test_size=0.2, random_state=42, stratify=y
)

# Standardisation x_std = (x - x_barre) / sigma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrainement de notre modele "From Scratch"
# Paramètres ajustables
model_home = ModelScratch(lr=0.3, iterations=1000)
model_home.fit(X_train_scaled, y_train)

# sklearn standard
model_sk_std = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
model_sk_std.fit(X_train_scaled, y_train)

# sklearn regularisé
# C=0.01 impose une forte pénalité Ridge pour stabiliser les poids
model_sk_reg = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
model_sk_reg.fit(X_train_scaled, y_train)

# keras (réseau de neurones)
# Normalisation (nécessaire pour ce type d'algorithme)
X_train_nn, X_test_nn = X_train / 16.0, X_test / 16.0
model_nn = Sequential([
    Input(shape=(64,)),
    Dense(16, activation="relu"),
    Dense(10, activation="softmax"),
])
model_nn.compile(
    optimizer=Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_nn.fit(X_train_nn, y_train, epochs=200, batch_size=32, verbose=0)

# analyse et affichage pour chaque model
def analyze_model(model, X, y, imgs, name, is_keras=False):
    if is_keras:
        probas = model.predict(X, verbose=0)
        y_pred = np.argmax(probas, axis=1)
    else:
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
        else:
            X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
            probas = model._softmax(X_bias @ model.theta.T)

    report(name, y, y_pred, probas)
    plot_confusion(y, y_pred, f"Matrice de Confusion : {name}")

    err_idx = np.where(y_pred != y)[0]
    show_examples(f"Erreurs : {name}", imgs, y, y_pred, err_idx)

    # ok_idx = np.where(y_pred == y)[0]
    # show_examples(f"Réussites : {name}", imgs, y, y_pred, ok_idx)

def theta_from_sklearn(clf):
    return np.hstack([clf.intercept_.reshape(-1, 1), clf.coef_])

def theta_from_keras_softmax(model):
    W, b = model.layers[-1].get_weights()
    return np.hstack([b.reshape(-1, 1), W.T])

analyze_model(model_home, X_test_scaled, y_test, img_test, "From scratch")
analyze_model(model_sk_std, X_test_scaled, y_test, img_test, "Sklearn Standard")
analyze_model(model_sk_reg, X_test_scaled, y_test, img_test, "Sklearn Régularisé (Ridge)")
analyze_model(model_nn, X_test_nn, y_test, img_test, "Keras", is_keras=True)

plot_coefficients(model_home.theta, "Poids par pixel : From scratch")
plot_coefficients(theta_from_sklearn(model_sk_std), "Poids par pixel : Sklearn (C=1.0)")
plot_coefficients(theta_from_sklearn(model_sk_reg), "Poids par pixel :  Sklearn (C=0.01)")

plt.show()