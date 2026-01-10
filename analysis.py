import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_coefficients(theta):
    weights = theta[:, 1:]

    plt.figure(figsize=(12, 5))
    plt.suptitle("Visualisation des coefficients appris (Poids par pixel)")

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        weight_image = weights[i].reshape(8, 8)
        plt.imshow(weight_image, cmap='RdBu_r', interpolation='nearest')
        plt.title(f"Chiffre {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title="Matrice de Confusion"):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)

