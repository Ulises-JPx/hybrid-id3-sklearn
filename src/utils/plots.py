
import matplotlib.pyplot as plt
import numpy as np
from .metrics import confusion_matrix_counts, per_class_metrics

def plot_confusion_matrix(true_labels, predicted_labels, output_filename, title="Confusion Matrix"):
    class_labels, matrix_counts = confusion_matrix_counts(true_labels, predicted_labels)
    confusion_matrix = np.array(matrix_counts, dtype=int)

    fig, ax = plt.subplots()
    image = ax.imshow(confusion_matrix)  # no explicit colormap per tool constraints
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(len(class_labels)):
        for col in range(len(class_labels)):
            ax.text(col, row, confusion_matrix[row, col], ha="center", va="center")

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def plot_per_class_metrics_bars(true_labels, predicted_labels, output_filename, title="Per-Class Metrics"):
    metrics_report = per_class_metrics(true_labels, predicted_labels)
    class_labels = metrics_report["classes"]

    precision_scores = [metrics_report["by_class"][label]["precision"] for label in class_labels]
    recall_scores    = [metrics_report["by_class"][label]["recall"]    for label in class_labels]
    f1_scores        = [metrics_report["by_class"][label]["f1"]        for label in class_labels]

    x_positions = np.arange(len(class_labels))
    bar_width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x_positions - bar_width, precision_scores, bar_width, label="Precision")
    ax.bar(x_positions,            recall_scores,    bar_width, label="Recall")
    ax.bar(x_positions + bar_width, f1_scores,       bar_width, label="F1")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def plot_accuracy_bar(accuracy_score, output_filename, title="Accuracy"):
    fig, ax = plt.subplots()
    ax.barh(["Accuracy"], [accuracy_score])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title(f"{title}: {accuracy_score:.3f}")
    ax.text(accuracy_score + 0.01, 0, f"{accuracy_score:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
