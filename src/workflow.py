"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file defines the main workflow for training, validating, and evaluating ID3 (entropy-based) decision trees using scikit-learn.
It provides functions for data normalization, splitting datasets into training and testing sets, training the decision tree model,
generating predictions, calculating metrics, saving results, and visualizing outputs such as confusion matrices and decision tree diagrams.
The workflow supports both showcase (train on all data) and validation (train/test split) modes, and ensures all results are saved
in a structured output directory for further analysis.
"""

import os
import csv
import json
from typing import List, Tuple, Dict
import numpy as np

from model.id3 import DecisionTreeID3Sklearn
from utils.files import save_text_to_file
from utils.metrics import accuracy, confusion_matrix_text, classification_report_text, classify_model
from utils.plots import plot_confusion_matrix, plot_per_class_metrics_bars, plot_accuracy_bar, plot_learning_curve
from utils.tree_viz import save_tree_png_safe

def clean_token(value):
    """
    Cleans and normalizes a single token value.

    Parameters:
        value: The input value to clean.

    Returns:
        The cleaned token as a string, or None if the token is empty or None.
    """
    if value is None:
        return None
    string_value = str(value).strip()
    # Remove surrounding quotes if present
    if len(string_value) >= 2 and ((string_value[0] == string_value[-1] == "'") or (string_value[0] == string_value[-1] == '"')):
        string_value = string_value[1:-1].strip()
    # Return None if the cleaned string is empty
    return string_value if string_value != "" else None

def normalize_table(features: List[List], targets: List) -> Tuple[List[List], List]:
    """
    Normalizes all feature and target values in the dataset.

    Parameters:
        features: List of feature rows (each row is a list of feature values).
        targets: List of target values.

    Returns:
        Tuple containing normalized features and targets.
    """
    normalized_features = [[clean_token(value) for value in row] for row in features]
    normalized_targets = [clean_token(target) for target in targets]
    return normalized_features, normalized_targets

def split_train_test(features: List[List], targets: List[str], train_ratio: float = 0.7, seed: int = 42) -> Tuple[List[List], List, List[List], List]:
    """
    Splits the dataset into training and testing sets, preserving class distribution.

    Parameters:
        features: List of feature rows.
        targets: List of target values.
        train_ratio: Proportion of data to use for training (default 0.7).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple containing training features, training targets, testing features, and testing targets.
    """
    test_ratio = 1.0 - float(train_ratio)
    rng = np.random.RandomState(seed)

    # Identify all unique classes in the target
    classes = sorted(set(targets))
    indices_by_class: Dict[str, List[int]] = {class_label: [] for class_label in classes}
    # Group indices by class for stratified splitting
    for index, class_label in enumerate(targets):
        indices_by_class[class_label].append(index)

    train_indices, test_indices = [], []
    # For each class, shuffle indices and split into train/test
    for class_label in classes:
        class_indices = indices_by_class[class_label][:]
        rng.shuffle(class_indices)
        num_samples = len(class_indices)
        # If only one sample, assign to training set
        if num_samples == 1:
            train_indices.extend(class_indices)
            continue
        num_test = int(round(test_ratio * num_samples))
        # Ensure at least one test sample and at least one train sample per class
        num_test = max(1, min(num_test, num_samples - 1))
        test_indices.extend(class_indices[:num_test])
        train_indices.extend(class_indices[num_test:])

    # Build train and test sets using selected indices
    train_features = [features[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]
    return train_features, train_targets, test_features, test_targets

def split_train_val_test(features, targets, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(features, targets, test_size=1-train_ratio, random_state=seed, stratify=targets)
    # split the remaining data into validation and test sets
    val_size = val_ratio / (1-train_ratio)  # proporcional dentro del temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_size, random_state=seed, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_train_val_test(feature_names: List[str],
                       features: List[List],
                       targets: List,
                       output_dir: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       seed: int = 42,
                       target_name: str = "target",
                       use_gridsearch: bool = False,
                       ccp_alpha: float = 0.0) -> Dict[str, float]:
    """
    Trains and evaluates the decision tree using a train/validation/test split.
    Saves splits, metadata, and results into the output directory.

    Returns:
        Dict with accuracies for validation and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize features and targets
    normalized_features, normalized_targets = normalize_table(features, targets)

    # Split data into train, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        normalized_features, normalized_targets,
        train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    # Prepare CSV headers
    csv_headers = list(feature_names) + [target_name]
    train_csv_path = os.path.join(output_dir, "train.csv")
    val_csv_path   = os.path.join(output_dir, "validation.csv")
    test_csv_path  = os.path.join(output_dir, "test.csv")

    # Save training data
    with open(train_csv_path, "w", newline="", encoding="utf-8") as train_file:
        writer = csv.writer(train_file)
        writer.writerow(csv_headers)
        for row, target in zip(X_train, y_train):
            writer.writerow(list(row) + [target])

    # Save validation data
    with open(val_csv_path, "w", newline="", encoding="utf-8") as val_file:
        writer = csv.writer(val_file)
        writer.writerow(csv_headers)
        for row, target in zip(X_val, y_val):
            writer.writerow(list(row) + [target])

    # Save testing data
    with open(test_csv_path, "w", newline="", encoding="utf-8") as test_file:
        writer = csv.writer(test_file)
        writer.writerow(csv_headers)
        for row, target in zip(X_test, y_test):
            writer.writerow(list(row) + [target])

    # Save metadata about split
    from collections import Counter
    metadata = {
        "split": {
            "ratio_train": train_ratio,
            "ratio_val": val_ratio,
            "ratio_test": 1.0 - train_ratio - val_ratio,
            "seed": seed
        },
        "target": target_name,
        "sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "class_counts": {
            "train": dict(Counter(y_train)),
            "val": dict(Counter(y_val)),
            "test": dict(Counter(y_test))
        }
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    # Train classifier
    classifier = DecisionTreeID3Sklearn(ccp_alpha=ccp_alpha)
    classifier.train(X_train, y_train, feature_names, use_gridsearch=use_gridsearch)

    # Validate
    val_preds = classifier.predict_batch(X_val)
    val_acc = accuracy(y_val, val_preds)

    # Test
    test_preds = classifier.predict_batch(X_test)
    test_acc = accuracy(y_test, test_preds)

    # Model classification summary (bias/variance/fit based on test set)
    bias = 1 - test_acc
    variance = np.var([1 if t == p else 0 for t, p in zip(y_test, test_preds)])
    classification = classify_model(bias, variance, test_acc)

    # === Learning Curve ===
    train_sizes = [int(len(X_train) * frac) for frac in [0.2, 0.4, 0.6, 0.8, 1.0]]
    train_scores = []
    val_scores = []

    for size in train_sizes:
        X_sub, y_sub = X_train[:size], y_train[:size]
        clf = DecisionTreeID3Sklearn()
        clf.train(X_sub, y_sub, feature_names, use_gridsearch=use_gridsearch)

        # Accuracy en train
        train_preds = clf.predict_batch(X_sub)
        train_scores.append(accuracy(y_sub, train_preds))

        # Accuracy en validation
        val_preds_sub = clf.predict_batch(X_val)
        val_scores.append(accuracy(y_val, val_preds_sub))

    # Save learning curve
    plot_learning_curve(
        train_sizes, train_scores, val_scores,
        output_filename=os.path.join(output_dir, "learning_curve.png")
    )

    with open(os.path.join(output_dir, "model_classification.txt"), "w", encoding="utf-8") as f:
        f.write(f"Bias: {classification['bias_level']}\n")
        f.write(f"Varianza: {classification['variance_level']}\n")
        f.write(f"Ajuste del modelo: {classification['fit_level']}\n")

    # Save all results (from test set)
    tree_text = classifier.print_tree()
    save_all_results(output_dir, y_test, test_preds, tree_text, test_acc,
                     confusion_matrix_title="Confusion Matrix (Test)",
                     metrics_title="Per-Class Metrics (Test)")

    return {"val_accuracy": val_acc, "test_accuracy": test_acc}


def save_all_results(output_dir: str, true_targets: List, predicted_targets: List, tree_text: str, accuracy_value: float, confusion_matrix_title: str, metrics_title: str):
    """
    Saves all results and visualizations to the specified output directory.

    Parameters:
        output_dir: Directory where results will be saved.
        true_targets: List of true target values.
        predicted_targets: List of predicted target values.
        tree_text: String representation of the decision tree.
        accuracy_value: Calculated accuracy of predictions.
        confusion_matrix_title: Title for confusion matrix plot.
        metrics_title: Title for per-class metrics plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Save decision tree text representation
    save_text_to_file(os.path.join(output_dir, "tree.txt"), tree_text)
    # Save decision tree visualization as PNG
    save_tree_png_safe(tree_text, os.path.join(output_dir, "tree.png"))

    # Generate and save classification report
    classification_report = classification_report_text(true_targets, predicted_targets)
    save_text_to_file(os.path.join(output_dir, "classification_report.txt"), classification_report)

    # Generate and save confusion matrix text
    confusion_matrix_str = confusion_matrix_text(true_targets, predicted_targets)
    save_text_to_file(os.path.join(output_dir, "confusion_matrix.txt"), confusion_matrix_str)
    # Plot and save confusion matrix image
    plot_confusion_matrix(true_targets, predicted_targets,
                          output_filename=os.path.join(output_dir, "confusion_matrix.png"),
                          title=confusion_matrix_title)

    # Plot and save per-class metrics bar chart
    plot_per_class_metrics_bars(true_targets, predicted_targets,
                                output_filename=os.path.join(output_dir, "per_class_metrics.png"),
                                title=metrics_title)

    # Plot and save accuracy bar chart
    plot_accuracy_bar(accuracy_value,
                      output_filename=os.path.join(output_dir, "accuracy.png"),
                      title="Accuracy")

def run_showcase(feature_names: List[str], features: List[List], targets: List, output_dir: str, tree_render=None, use_gridsearch: bool = False, ccp_alpha: float = 0.0) -> float:
    """
    Trains and evaluates the decision tree on the entire dataset (showcase mode).

    Parameters:
        feature_names: List of feature names.
        features: List of feature rows.
        targets: List of target values.
        output_dir: Directory to save results.
        tree_render: Optional tree rendering configuration (unused).
        use_gridsearch: Whether to use grid search for hyperparameter tuning (default False).

    Returns:
        Accuracy of predictions on the entire dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Normalize features and targets
    normalized_features, normalized_targets = normalize_table(features, targets)
    # Initialize and train the decision tree classifier
    classifier = DecisionTreeID3Sklearn(ccp_alpha=ccp_alpha)
    classifier.train(normalized_features, normalized_targets, feature_names, use_gridsearch=use_gridsearch, )

    # Generate predictions on the entire dataset
    predictions = classifier.predict_batch(normalized_features)
    # Calculate accuracy
    accuracy_value = accuracy(normalized_targets, predictions)

    bias = 1 - accuracy_value  # ejemplo simple: bias ~ error
    variance = np.var([1 if t == p else 0 for t, p in zip(normalized_targets, predictions)])
    classification = classify_model(bias, variance, accuracy_value)

    classification_path = os.path.join(output_dir, "model_classification.txt")
    with open(classification_path, "w", encoding="utf-8") as f:
        f.write(f"Bias: {classification['bias_level']}\n")
        f.write(f"Varianza: {classification['variance_level']}\n")
        f.write(f"Ajuste del modelo: {classification['fit_level']}\n")

    # Get string representation of the trained tree
    tree_text = classifier.print_tree()
    # Save all results and visualizations
    save_all_results(output_dir, normalized_targets, predictions, tree_text, accuracy_value,
                     confusion_matrix_title="Confusion Matrix (Train/Showcase)",
                     metrics_title="Per-Class Metrics (Train/Showcase)")
    return accuracy_value

def run_validation(feature_names: List[str],
                   features: List[List],
                   targets: List,
                   output_dir: str,
                   train_ratio: float = 0.7,
                   seed: int = 42,
                   tree_render=None,
                   target_name: str = "target",
                   use_gridsearch: bool = False,
                   ccp_alpha: float = 0.0) -> float:
    """
    Trains and evaluates the decision tree using a train/test split (validation mode).

    Parameters:
        feature_names: List of feature names.
        features: List of feature rows.
        targets: List of target values.
        output_dir: Directory to save results.
        train_ratio: Proportion of data to use for training (default 0.7).
        seed: Random seed for reproducibility (default 42).
        tree_render: Optional tree rendering configuration (unused).
        target_name: Name of the target column (default "target").
        use_gridsearch: Whether to use grid search for hyperparameter tuning (default False).

    Returns:
        Accuracy of predictions on the test set.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize features and targets
    normalized_features, normalized_targets = normalize_table(features, targets)
    # Split data into training and testing sets
    train_features, train_targets, test_features, test_targets = split_train_test(normalized_features, normalized_targets, train_ratio=train_ratio, seed=seed)

    # Prepare CSV headers for saving train/test splits
    csv_headers = list(feature_names) + [target_name]
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path  = os.path.join(output_dir, "test.csv")

    # Save training data to CSV
    with open(train_csv_path, "w", newline="", encoding="utf-8") as train_file:
        writer = csv.writer(train_file)
        writer.writerow(csv_headers)
        for row, target in zip(train_features, train_targets):
            writer.writerow(list(row) + [target])

    # Save testing data to CSV
    with open(test_csv_path, "w", newline="", encoding="utf-8") as test_file:
        writer = csv.writer(test_file)
        writer.writerow(csv_headers)
        for row, target in zip(test_features, test_targets):
            writer.writerow(list(row) + [target])

    # Collect metadata about the split and save as JSON
    from collections import Counter
    metadata = {
        "split": {"ratio_train": train_ratio, "ratio_test": 1.0 - train_ratio, "seed": seed},
        "target": target_name,
        "sizes": {"train": len(train_features), "test": len(test_features)},
        "class_counts": {"train": dict(Counter(train_targets)), "test": dict(Counter(test_targets))}
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    # Initialize and train the decision tree classifier on training data
    classifier = DecisionTreeID3Sklearn(ccp_alpha=ccp_alpha)
    classifier.train(train_features, train_targets, feature_names, use_gridsearch=use_gridsearch)

    # Generate predictions on the test set
    predictions = classifier.predict_batch(test_features)
    # Calculate accuracy on the test set
    accuracy_value = accuracy(test_targets, predictions)

    bias = 1 - accuracy_value
    variance = np.var([1 if t == p else 0 for t, p in zip(test_targets, predictions)])
    classification = classify_model(bias, variance, accuracy_value)

    classification_path = os.path.join(output_dir, "model_classification.txt")
    with open(classification_path, "w", encoding="utf-8") as f:
        f.write(f"Bias: {classification['bias_level']}\n")
        f.write(f"Varianza: {classification['variance_level']}\n")
        f.write(f"Ajuste del modelo: {classification['fit_level']}\n")

    # Get string representation of the trained tree
    tree_text = classifier.print_tree()
    # Save all results and visualizations for the test set
    save_all_results(output_dir, test_targets, predictions, tree_text, accuracy_value,
                     confusion_matrix_title="Confusion Matrix (Test/Validation)",
                     metrics_title="Per-Class Metrics (Test/Validation)")
    return accuracy_value