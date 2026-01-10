import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(int)

X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Dense(16, activation="relu", input_shape=(64,)),
    Dense(10, activation="softmax")
])
model.compile(optimizer=Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(proba, axis=1)

print("Accuracy test:", accuracy_score(y_test, y_pred))
print("Log loss test:", log_loss(y_test, proba, labels=list(range(10))))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))