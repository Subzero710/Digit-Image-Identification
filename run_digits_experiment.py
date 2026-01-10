import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from softmax_regression_from_scratch import SoftmaxRegressionGD, add_bias

TEST_SIZE = 0.2
SEED = 0

LR = 0.1
EPOCHS = 5000
TOL = 1e-6
L2 = 0.0
PRINT_EVERY = 200

def per_class_acc(cm):
    denom = cm.sum(axis=1)
    denom[denom == 0] = 1
    return np.diag(cm) / denom

def evaluate(name, y_true, y_pred, proba):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    print(f"\n--- {name} ---")
    print(f"accuracy  = {accuracy_score(y_true, y_pred):.4f}")
    print(f"log_loss  = {log_loss(y_true, proba, labels=np.arange(10)):.4f}")
    print("per-class accuracy =", np.round(per_class_acc(cm), 4))
    print("\nMatrice de confusion:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

def main():
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr_b = add_bias(X_tr)
    X_te_b = add_bias(X_te)

    scratch = SoftmaxRegressionGD(lr=LR, epochs=EPOCHS, tol=TOL, l2=L2, seed=SEED, print_every=PRINT_EVERY, verbose=True)
    scratch.fit(X_tr_b, y_tr, C=10)

    p_s = scratch.predict_proba(X_te_b)
    y_s = p_s.argmax(axis=1)
    evaluate("from_scratch_softmax", y_te, y_s, p_s)

    sk = LogisticRegression(max_iter=20000)
    sk.fit(X_tr, y_tr)
    p_k = sk.predict_proba(X_te)
    y_k = p_k.argmax(axis=1)
    evaluate("sklearn_logreg", y_te, y_k, p_k)

if __name__ == "__main__":
    main()