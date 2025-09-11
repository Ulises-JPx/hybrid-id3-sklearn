"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file provides utility functions for managing result directories and handling file operations
within the hybrid-id3-sklearn project. It includes functionality to ensure the existence of specific
result directories, clear their contents while preserving README files, reset directories to a clean
state, save text data to files, and load CSV data for further processing.
"""

import csv
import os
import shutil

def ensure_result_directories(base_results_path: str):
    """
    Ensures that the required result directories ('showcase' and 'validation') exist within the
    specified base results path. If the directories do not exist, they are created.

    Parameters:
        base_results_path (str): The root directory where result subdirectories should be located.
    """
    # Create the 'showcase' directory if it does not exist
    os.makedirs(os.path.join(base_results_path, "showcase"), exist_ok=True)
    # Create the 'validation' directory if it does not exist
    os.makedirs(os.path.join(base_results_path, "validation"), exist_ok=True)

def clear_directory_contents(directory_path: str):
    """
    Removes all files and subdirectories within the specified directory, except for any file named
    'README.md' (case-insensitive). If the directory does not exist, it is created.

    Parameters:
        directory_path (str): The path to the directory whose contents should be cleared.
    """
    # If the directory does not exist, create it and exit
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return

    # Iterate through all items in the directory
    for item_name in os.listdir(directory_path):
        # Skip the README.md file to preserve documentation
        if item_name.lower() == "readme.md":
            continue

        item_path = os.path.join(directory_path, item_name)

        # Remove files and symbolic links
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        # Remove subdirectories and their contents recursively
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def reset_result_directories(base_results_path: str):
    """
    Ensures that the result directories exist and clears their contents, except for README files.
    This function is useful for resetting the state of result directories before running new
    experiments or validations.

    Parameters:
        base_results_path (str): The root directory containing the result subdirectories.
    """
    # Ensure that both 'showcase' and 'validation' directories exist
    ensure_result_directories(base_results_path)
    # Clear contents of the 'showcase' directory
    clear_directory_contents(os.path.join(base_results_path, "showcase"))
    # Clear contents of the 'validation' directory
    clear_directory_contents(os.path.join(base_results_path, "validation"))

def save_text_to_file(file_path: str, text_content: str):
    """
    Saves the provided text content to a file at the specified path. If the parent directory does
    not exist, it is created. If the text content is None, an empty string is written.

    Parameters:
        file_path (str): The full path to the file where the text should be saved.
        text_content (str): The text content to write to the file.
    """
    # Ensure the parent directory exists before writing the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Open the file for writing with UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as file:
        # Write the provided text content, or an empty string if None
        file.write(text_content if text_content is not None else "")

def load_csv_data(csv_file_path: str):
    """
    Loads data from a CSV file and separates it into features and target values. Assumes that the
    last column in each row is the target variable, and all preceding columns are features.

    Parameters:
        csv_file_path (str): The path to the CSV file to be loaded.

    Returns:
        tuple: A tuple containing:
            - header (list of str): The column names for the features.
            - features (list of list of str): The feature values for each row.
            - targets (list of str): The target values for each row.
    """
    # Open the CSV file for reading
    with open(csv_file_path, "r", newline="") as file:
        # Read all rows from the CSV file
        rows = list(csv.reader(file))

    # Extract the header row (column names)
    header = rows[0]
    # Extract all data rows (excluding the header)
    data_rows = rows[1:]

    # Separate features and targets for each row
    features = [row[:-1] for row in data_rows]  # All columns except the last are features
    targets = [row[-1] for row in data_rows]    # The last column is the target

    # Return the header (excluding the target column), features, and targets
    return header[:-1], features, targets
