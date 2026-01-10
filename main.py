import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from model_scratch import LogitScratch
from analysis import plot_coefficients, plot_confusion
from analyze_digits_results import report, show_examples

#preparation des outils
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, digits.images, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#entrainement
model_home = LogitScratch(lr=0.1, iterations=1000)
model_home.fit(X_train_scaled, y_train)

#sklearn standard
model_sk_std = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
model_sk_std.fit(X_train_scaled, y_train)

#sklearn standard
model_sk_reg = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
model_sk_reg.fit(X_train_scaled, y_train)

# keras
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
model_nn.fit(X_train_nn, y_train, epochs=20, batch_size=32, verbose=0)

#analyse
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
    plot_confusion(y, y_pred, f"Confusion : {name}")

    err_idx = np.where(y_pred != y)[0]
    print(f"\nExemples d'erreurs : {name}")
    for i in err_idx[:12]:
        print(f"guess = {y_pred[i]}, real = {y[i]}")
    show_examples(f"Erreurs : {name}", imgs, y, y_pred, err_idx)

    ok_idx = np.where(y_pred == y)[0]
    print(f"\nExemples de réussites : {name}")
    for i in ok_idx[:12]:
        print(f"guess = {y_pred[i]}, real = {y[i]}")
    show_examples(f"Réussites : {name}", imgs, y, y_pred, ok_idx)

#lancement des analyses pour chaque modèle
analyze_model(model_home, X_test_scaled, y_test, img_test, "From scratch")
analyze_model(model_sk_std, X_test_scaled, y_test, img_test, "Sklearn Standard")
analyze_model(model_sk_reg, X_test_scaled, y_test, img_test, "Sklearn Régularisé (L2)")
analyze_model(model_nn, X_test_nn, y_test, img_test, "Keras", is_keras=True)
plot_coefficients(model_home.theta)
plt.show()