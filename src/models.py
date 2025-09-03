import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)
from sklearn.svm import SVC
import numpy as np
import os

def binarize_labels(y, threshold=5):
    return np.where(y >= threshold, 1, 0)

def train_and_evaluate(X, y, results_dir="results"):
    y = binarize_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    os.makedirs(results_dir, exist_ok=True)
    fig_cm = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(fig_cm)
    print(f"Confusion matrix saved to {fig_cm}")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    fig_roc = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(fig_roc)
    print(f"ROC curve saved to {fig_roc}")
    plt.close()

    report = classification_report(y_test, y_pred, digits=4)
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    print(f"Metrics saved to {metrics_path}")
