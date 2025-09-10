
from typing import List
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

def accuracy(true_labels: List[str], predicted_labels: List[str]) -> float:
    return accuracy_score(true_labels, predicted_labels)

def confusion_matrix_counts(true_labels: List[str], predicted_labels: List[str]):
    matrix = confusion_matrix(true_labels, predicted_labels, labels=sorted(set(true_labels) | set(predicted_labels)))
    classes = sorted(set(true_labels) | set(predicted_labels))
    return classes, matrix

def confusion_matrix_text(true_labels: List[str], predicted_labels: List[str]) -> str:
    classes, matrix = confusion_matrix_counts(true_labels, predicted_labels)
    header = ["True\\Pred"] + classes
    lines = ["\t".join(header)]
    for i, class_label in enumerate(classes):
        row = [str(class_label)] + [str(matrix[i, j]) for j in range(len(classes))]
        lines.append("\t".join(row))
    return "\n".join(lines)

def classification_report_text(true_labels: List[str], predicted_labels: List[str]) -> str:
    return classification_report(true_labels, predicted_labels, digits=3, zero_division=0)

def per_class_metrics(true_labels: List[str], predicted_labels: List[str]):
    from sklearn.metrics import precision_recall_fscore_support
    classes = sorted(set(true_labels) | set(predicted_labels))
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, labels=classes, zero_division=0)
    by_class = {}
    for idx, cls in enumerate(classes):
        by_class[cls] = {
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': support[idx]
        }
    return {
        'classes': classes,
        'by_class': by_class
    }
