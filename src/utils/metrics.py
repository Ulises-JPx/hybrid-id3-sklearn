"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file provides a set of utility functions for evaluating classification models.
It includes functions to compute accuracy, generate confusion matrices (both as raw counts and formatted text),
produce detailed classification reports, and calculate per-class metrics such as precision, recall, F1-score, and support.
All functions are designed to work with lists of true and predicted labels, and leverage scikit-learn's metrics for robust evaluation.
"""

from typing import List
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

def accuracy(true_labels: List[str], predicted_labels: List[str]) -> float:
    """
    Calculates the overall accuracy of predictions.

    Parameters:
        true_labels (List[str]): The list of actual class labels.
        predicted_labels (List[str]): The list of predicted class labels.

    Returns:
        float: The accuracy score, representing the proportion of correct predictions.
    """
    # Compute the accuracy using scikit-learn's accuracy_score function.
    return accuracy_score(true_labels, predicted_labels)

def confusion_matrix_counts(true_labels: List[str], predicted_labels: List[str]):
    """
    Generates the confusion matrix as a numpy array along with the sorted list of class labels.

    Parameters:
        true_labels (List[str]): The list of actual class labels.
        predicted_labels (List[str]): The list of predicted class labels.

    Returns:
        Tuple[List[str], numpy.ndarray]: A tuple containing the sorted list of class labels and the confusion matrix.
    """
    # Determine all unique classes present in either true or predicted labels.
    all_classes = sorted(set(true_labels) | set(predicted_labels))
    # Compute the confusion matrix using the determined classes.
    matrix = confusion_matrix(true_labels, predicted_labels, labels=all_classes)
    return all_classes, matrix

def confusion_matrix_text(true_labels: List[str], predicted_labels: List[str]) -> str:
    """
    Produces a formatted string representation of the confusion matrix for easier visualization.

    Parameters:
        true_labels (List[str]): The list of actual class labels.
        predicted_labels (List[str]): The list of predicted class labels.

    Returns:
        str: A tab-separated string representing the confusion matrix with headers.
    """
    # Retrieve the classes and confusion matrix counts.
    classes, matrix = confusion_matrix_counts(true_labels, predicted_labels)
    # Prepare the header row for the matrix.
    header = ["True\\Pred"] + classes
    # Initialize the list of lines with the header.
    lines = ["\t".join(header)]
    # Iterate through each class to build the matrix rows.
    for i, class_label in enumerate(classes):
        # Construct each row with the class label and corresponding counts.
        row = [str(class_label)] + [str(matrix[i, j]) for j in range(len(classes))]
        lines.append("\t".join(row))
    # Join all lines into a single string separated by newlines.
    return "\n".join(lines)

def classification_report_text(true_labels: List[str], predicted_labels: List[str]) -> str:
    """
    Generates a detailed classification report including precision, recall, F1-score, and support for each class.

    Parameters:
        true_labels (List[str]): The list of actual class labels.
        predicted_labels (List[str]): The list of predicted class labels.

    Returns:
        str: A formatted string containing the classification report.
    """
    # Use scikit-learn's classification_report to generate the report.
    # digits=3 ensures three decimal places; zero_division=0 avoids division errors.
    return classification_report(true_labels, predicted_labels, digits=3, zero_division=0)

def per_class_metrics(true_labels: List[str], predicted_labels: List[str]):
    """
    Computes precision, recall, F1-score, and support for each class individually.

    Parameters:
        true_labels (List[str]): The list of actual class labels.
        predicted_labels (List[str]): The list of predicted class labels.

    Returns:
        dict: A dictionary containing:
            - 'classes': The sorted list of class labels.
            - 'by_class': A dictionary mapping each class label to its metrics (precision, recall, F1-score, support).
    """
    # Determine all unique classes present in either true or predicted labels.
    all_classes = sorted(set(true_labels) | set(predicted_labels))
    # Calculate per-class metrics using scikit-learn's precision_recall_fscore_support.
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=all_classes,
        zero_division=0
    )
    # Build a dictionary mapping each class to its metrics.
    metrics_by_class = {}
    for idx, class_label in enumerate(all_classes):
        metrics_by_class[class_label] = {
            'precision': precision[idx],
            'recall': recall[idx],
            'f1': f1[idx],
            'support': support[idx]
        }
    # Return the classes and the metrics dictionary.
    return {
        'classes': all_classes,
        'by_class': metrics_by_class
    }

def classify_model(bias, variance, accuracy, threshold_low=0.33, threshold_high=0.66):
    """
    Clasifica bias, varianza y ajuste del modelo en categorías cualitativas.
    :param bias: valor numérico del bias (0 a 1 recomendado)
    :param variance: valor numérico de la varianza (0 a 1 recomendado)
    :param accuracy: exactitud del modelo (0 a 1)
    :return: diccionario con bias_level, variance_level y fit_level
    """

    # --- Bias ---
    if bias < threshold_low:
        bias_level = "Bajo"
    elif bias < threshold_high:
        bias_level = "Medio"
    else:
        bias_level = "Alto"

    # --- Varianza ---
    if variance < threshold_low:
        variance_level = "Bajo"
    elif variance < threshold_high:
        variance_level = "Medio"
    else:
        variance_level = "Alto"

    # --- Nivel de ajuste ---
    # regla simple: underfitting = high bias + low variance
    # overfitting = low bias + high variance
    # fit = lo demás
    if bias_level == "Alto" and variance_level == "Bajo":
        fit_level = "Underfit"
    elif bias_level == "Bajo" and variance_level == "Alto":
        fit_level = "Overfit"
    else:
        fit_level = "Fit"

    return {
        "bias_level": bias_level,
        "variance_level": variance_level,
        "fit_level": fit_level
    }

