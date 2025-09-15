"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file provides utility functions for generating and saving various classification performance plots using matplotlib.
It includes functions to plot a confusion matrix, per-class metric bar charts (precision, recall, F1-score), and a horizontal bar for overall accuracy.
These visualizations are useful for evaluating and presenting the results of classification models.
All functions accept true and predicted labels, compute the necessary metrics, and save the resulting plots to the specified output file.
"""

import matplotlib.pyplot as plt
import numpy as np
from .metrics import confusion_matrix_counts, per_class_metrics

def plot_confusion_matrix(true_labels, predicted_labels, output_filename, title="Confusion Matrix"):
    """
    Plots and saves a confusion matrix for the given true and predicted labels.

    Parameters:
        true_labels (list or array): The ground truth class labels.
        predicted_labels (list or array): The predicted class labels from the classifier.
        output_filename (str): Path to save the generated confusion matrix plot.
        title (str): Title for the plot (default: "Confusion Matrix").
    """
    # Compute class labels and confusion matrix counts using the provided utility function
    class_labels, matrix_counts = confusion_matrix_counts(true_labels, predicted_labels)
    # Convert the confusion matrix counts to a NumPy array for plotting
    confusion_matrix = np.array(matrix_counts, dtype=int)

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()
    # Display the confusion matrix as an image with a blue color map
    image = ax.imshow(confusion_matrix, cmap="Blues")

    # Set the x and y axis ticks and labels to correspond to the class labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    # Label the axes and set the plot title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell in the confusion matrix with its count value
    for row in range(len(class_labels)):
        for col in range(len(class_labels)):
            ax.text(col, row, confusion_matrix[row, col], ha="center", va="center")

    # Add a colorbar to indicate the scale of values in the matrix
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    # Save the plot to the specified output file
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the plot to free up resources
    plt.close()

def plot_per_class_metrics_bars(true_labels, predicted_labels, output_filename, title="Per-Class Metrics"):
    """
    Plots and saves a grouped bar chart for per-class precision, recall, and F1-score.

    Parameters:
        true_labels (list or array): The ground truth class labels.
        predicted_labels (list or array): The predicted class labels from the classifier.
        output_filename (str): Path to save the generated bar chart.
        title (str): Title for the plot (default: "Per-Class Metrics").
    """
    # Compute per-class metrics using the provided utility function
    metrics_report = per_class_metrics(true_labels, predicted_labels)
    class_labels = metrics_report["classes"]

    # Extract precision, recall, and F1-score for each class
    precision_scores = [metrics_report["by_class"][label]["precision"] for label in class_labels]
    recall_scores    = [metrics_report["by_class"][label]["recall"]    for label in class_labels]
    f1_scores        = [metrics_report["by_class"][label]["f1"]        for label in class_labels]

    # Set positions for the grouped bars
    x_positions = np.arange(len(class_labels))
    bar_width = 0.25

    # Create a new figure and axis for the bar chart
    fig, ax = plt.subplots()
    # Plot precision bars shifted to the left
    ax.bar(x_positions - bar_width, precision_scores, bar_width, label="Precision")
    # Plot recall bars centered
    ax.bar(x_positions,            recall_scores,    bar_width, label="Recall")
    # Plot F1-score bars shifted to the right
    ax.bar(x_positions + bar_width, f1_scores,       bar_width, label="F1")

    # Set x-axis ticks and labels to class names, rotated for readability
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    # Set y-axis limits to cover the range of metric scores
    ax.set_ylim(0, 1.05)
    # Label the y-axis and set the plot title
    ax.set_ylabel("Score")
    ax.set_title(title)
    # Add a legend to distinguish the metrics
    ax.legend(loc="lower right")

    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    # Save the plot to the specified output file
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the plot to free up resources
    plt.close()

def plot_accuracy_bar(accuracy_score, output_filename, title="Accuracy"):
    """
    Plots and saves a horizontal bar chart representing the overall accuracy score.

    Parameters:
        accuracy_score (float): The overall accuracy of the classifier (between 0 and 1).
        output_filename (str): Path to save the generated accuracy bar chart.
        title (str): Title for the plot (default: "Accuracy").
    """
    # Create a new figure and axis for the horizontal bar chart
    fig, ax = plt.subplots()
    # Plot a single horizontal bar for accuracy
    ax.barh(["Accuracy"], [accuracy_score])
    # Set the x-axis limits to cover the full range of possible accuracy values
    ax.set_xlim(0, 1.0)
    # Label the x-axis and set the plot title, including the accuracy value
    ax.set_xlabel("Score")
    ax.set_title(f"{title}: {accuracy_score:.3f}")
    # Annotate the bar with the accuracy value for clarity
    ax.text(accuracy_score + 0.01, 0, f"{accuracy_score:.3f}", va="center")
    # Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()
    # Save the plot to the specified output file
    plt.savefig(output_filename, bbox_inches="tight")
    # Close the plot to free up resources
    plt.close()

def plot_learning_curve(train_sizes, train_scores, val_scores, output_filename="learning_curve.png"):
    """
    Plots and saves a learning curve showing training and validation accuracy vs training size.

    Parameters:
        train_sizes (list): List of training set sizes.
        train_scores (list): Accuracy scores for training sets.
        val_scores (list): Accuracy scores for validation sets.
        output_filename (str): Path to save the plot (default: "learning_curve.png").
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_scores, marker="s", label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def plot_accuracy_comparison(showcase_acc, validation_acc, val_acc, test_acc, output_filename="accuracy_comparison.png"):
    """
    Plots and saves a bar chart comparing accuracies from different evaluation modes.

    Parameters:
        showcase_acc (float): Accuracy using all data (showcase mode).
        validation_acc (float): Accuracy from train/test split (validation mode).
        val_acc (float): Validation accuracy from train/val/test split.
        test_acc (float): Test accuracy from train/val/test split.
        output_filename (str): Path to save the plot (default: "accuracy_comparison.png").
    """
    labels = ["Showcase", "Validation (Test)", "Train/Val (Val)", "Train/Val (Test)"]
    values = [showcase_acc, validation_acc, val_acc, test_acc]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=["#4CAF50", "#2196F3", "#FFC107", "#F44336"])
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Comparison of Model Accuracies")
    plt.gca().set_title("Comparison of Model Accuracies", pad=20)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
