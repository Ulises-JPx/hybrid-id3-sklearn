
"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This script serves as the main entry point for running machine learning experiments
using the ID3 algorithm. It provides automatic detection of the target column in a
CSV dataset, with the option to override the selection if ambiguity arises.
"""

import os
import sys
import argparse
from typing import List, Tuple
import pandas as pd

from utils.files import reset_result_directories, ensure_result_directories
from utils.plots import plot_accuracy_comparison
from workflow import run_showcase, run_validation, run_train_val_test

# Fixed configuration for train/test split ratio and random seed
TRAIN_TEST_RATIO = 0.8
RANDOM_SEED = 42

def clean_cell_token(value):
    """
    Cleans a single cell value from the dataset.
    """
    if value is None:
        return None
    string_value = str(value).strip()
    if len(string_value) >= 2 and ((string_value[0] == string_value[-1] == "'") or (string_value[0] == string_value[-1] == '"')):
        string_value = string_value[1:-1].strip()
    return string_value if string_value != "" else None

# Set of known target column names for automatic detection
KNOWN_TARGET_COLUMN_NAMES = {"class", "target", "label", "y", "outcome", "diagnosis"}

def auto_detect_target_column(dataframe: pd.DataFrame) -> Tuple[str, str, List[str]]:
    """
    Automatically detects the most plausible target column in the dataset.
    """
    column_names = list(dataframe.columns)
    num_rows = len(dataframe)

    # 1) Known target names
    for name in column_names:
        if name.lower() in KNOWN_TARGET_COLUMN_NAMES:
            return name, "recognized name", [name]

    # 2) Low-cardinality candidates
    candidates = []
    for column in column_names:
        unique_values = dataframe[column].nunique(dropna=True)
        if 2 <= unique_values <= min(20, max(2, num_rows // 10)) and unique_values < 0.5 * max(1, num_rows):
            candidates.append((column, unique_values))
    if candidates:
        candidates.sort(key=lambda x: (x[1], column_names.index(x[0])))
        top_candidate = candidates[0][0]
        return top_candidate, f"low cardinality (unique={candidates[0][1]})", [col for col, _ in candidates[:6]]

    # 3) Fallback: last column
    return column_names[-1], "last column by convention", [column_names[-1]]

def load_csv_and_select_target(csv_path: str, target_column: str = None) -> Tuple[List[str], List[List[str]], List[str], pd.DataFrame]:
    """
    Loads a CSV file, cleans cell values, and separates features and target.
    """
    dataframe = pd.read_csv(csv_path, dtype=str)
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(clean_cell_token)

    column_names = list(dataframe.columns)
    if target_column is None:
        target_column, reason, candidates = auto_detect_target_column(dataframe)
        print(f"[auto] Selected target column: '{target_column}' ({reason}).")
        if len(candidates) > 1:
            print(f"[auto] Other candidates: {candidates[1:]}")
    else:
        if target_column not in column_names:
            raise ValueError(f"Target column '{target_column}' does not exist. Available columns: {column_names}")

    target_values = dataframe[target_column].tolist()
    features_dataframe = dataframe.drop(columns=[target_column])
    feature_names = features_dataframe.columns.tolist()
    feature_rows = features_dataframe.values.tolist()
    return feature_names, feature_rows, target_values, dataframe

def list_csv_files_in_directory(directory_path: str) -> List[str]:
    if not os.path.isdir(directory_path):
        return []
    return sorted([
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if filename.lower().endswith(".csv")
    ])

def prompt_user_choice(prompt_message: str, options: List[str], default_index: int = 0) -> str:
    print(prompt_message)
    for idx, option in enumerate(options, 1):
        print(f"  {idx}) {option}")
    user_input = input(f"Choose [1-{len(options)}] (Enter={default_index+1}): ").strip()
    if user_input == "":
        return options[default_index]
    try:
        selected_index = int(user_input)
        if 1 <= selected_index <= len(options):
            return options[selected_index - 1]
    except ValueError:
        pass
    print("Invalid input, using default option.")
    return options[default_index]

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="ID3 Experiments — automatic target selection with override if ambiguous."
    )
    parser.add_argument("--data", "-d", type=str, default=None,
                        help="Path to the CSV dataset. If not provided, a menu will show ./data/*.csv files.")
    parser.add_argument("--target", "-t", type=str, default=None,
                        help="Force the target column (if provided, it will be used as is).")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base directory for results (will be cleaned on each run).")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Avoid prompts even if there is ambiguity in target selection.")
    parser.add_argument("--gridsearch", action="store_true", help="Activa GridSearchCV para optimización de hiperparámetros.")
    return parser

def main():
    args = build_argument_parser().parse_args()

    # Step 1: Dataset selection
    dataset_path = args.data
    if dataset_path is None and not args.no_interactive:
        available_datasets = list_csv_files_in_directory("data")
        if not available_datasets:
            print("No CSV files found in ./data. Please specify the path using --data.")
            sys.exit(1)
        dataset_path = prompt_user_choice("Select the dataset:", available_datasets, default_index=0)
    if dataset_path is None:
        print("You must specify --data when using --no-interactive.")
        sys.exit(1)
    if not os.path.isfile(dataset_path):
        print(f"File does not exist: {dataset_path}")
        sys.exit(1)

    # Step 2: Target detection
    preview_dataframe = pd.read_csv(dataset_path, dtype=str)
    suggested_target, detection_reason, candidate_targets = auto_detect_target_column(preview_dataframe)
    selected_target_column = args.target or suggested_target

    if (args.target is None) and (len(candidate_targets) > 1) and (not args.no_interactive):
        default_index = candidate_targets.index(suggested_target) if suggested_target in candidate_targets else 0
        selected_target_column = prompt_user_choice(
            f"\nMultiple plausible target columns detected (auto={suggested_target}, {detection_reason}). Please choose one:",
            candidate_targets,
            default_index=default_index
        )

    # Step 2.5: GridSearchCV option
    use_gridsearch = args.gridsearch
    if not args.no_interactive:
        gridsearch_options = ["No (más rápido)", "Sí (optimiza hiperparámetros)"]
        default_gridsearch = 0 if not use_gridsearch else 1
        user_choice = prompt_user_choice(
            "¿Deseas activar GridSearchCV para optimización de hiperparámetros?",
            gridsearch_options,
            default_index=default_gridsearch
        )
        use_gridsearch = (user_choice == gridsearch_options[1])

    # Step 2.6: Cost-Complexity Pruning option
    ccp_alpha = 0.0
    if not args.no_interactive:
        pruning_options = [
            "No (sin poda) -> ccp_alpha=0.0",
            "Poda ligera -> ccp_alpha=0.001",
            "Poda media -> ccp_alpha=0.01",
            "Poda fuerte -> ccp_alpha=0.1"
        ]
        user_choice = prompt_user_choice(
            "¿Deseas activar Cost-Complexity Pruning (poda)?",
            pruning_options,
            default_index=0
        )
        if user_choice == pruning_options[1]:
            ccp_alpha = 0.001
        elif user_choice == pruning_options[2]:
            ccp_alpha = 0.01
        elif user_choice == pruning_options[3]:
            ccp_alpha = 0.1



    # Step 3: Load cleaned dataset with selected target
    try:
        feature_names, feature_rows, target_values, cleaned_dataframe = load_csv_and_select_target(
            dataset_path, target_column=selected_target_column
        )
    except Exception as error:
        print(f"Error loading/selecting target column: {error}")
        sys.exit(1)

    # Small run header
    print(f"\n[run] Dataset: {dataset_path}")
    print(f"[run] Target : {selected_target_column}")
    print(f"[run] GridSearchCV: {'Activado' if use_gridsearch else 'Desactivado'}")

    # Step 4: Reset results dir
    reset_result_directories(args.results_dir)
    ensure_result_directories(args.results_dir)
    showcase_results_directory = os.path.join(args.results_dir, "showcase")
    validation_results_directory = os.path.join(args.results_dir, "validation")
    train_val_test_results_directory = os.path.join(args.results_dir, "train_val_test")


    # Step 5: Tree render config (kept for compatibility)
    TREE_RENDER_CONFIGURATION = {
        "max_dim_px": 64000,
        "font_size": 12,
        "dpi": 200,
        "padding_px": 24,
    }

    # Step 6: Showcase
    training_accuracy = run_showcase(
        feature_names, feature_rows, target_values, showcase_results_directory,
        tree_render=TREE_RENDER_CONFIGURATION,
        use_gridsearch=use_gridsearch,
        ccp_alpha=ccp_alpha
    )
    print("\n=== SHOWCASE ===")
    print("** Training and testing on the entire dataset **\n")
    print(f"Training accuracy = {training_accuracy:.4f} (results saved in {showcase_results_directory})\n")

    # Mostrar clasificación del modelo
    classification_file = os.path.join(showcase_results_directory, "model_classification.txt")
    if os.path.exists(classification_file):
        with open(classification_file, "r", encoding="utf-8") as f:
            print(f.read())

    # Step 7: Validation (pass target_name to label the CSV headers correctly)
    test_accuracy = run_validation(
        feature_names, feature_rows, target_values, validation_results_directory,
        train_ratio=TRAIN_TEST_RATIO, seed=RANDOM_SEED,
        tree_render=TREE_RENDER_CONFIGURATION,
        target_name=selected_target_column,
        use_gridsearch=use_gridsearch,
        ccp_alpha=ccp_alpha
    )
    print("\n=== VALIDATION ===")
    print(f"** Training/testing split: {int(TRAIN_TEST_RATIO*100)}/{100-int(TRAIN_TEST_RATIO*100)}, seed={RANDOM_SEED} **\n")
    print(f"Test accuracy = {test_accuracy:.4f} (results saved in {validation_results_directory})\n")

    # Mostrar clasificación del modelo
    classification_file = os.path.join(validation_results_directory, "model_classification.txt")
    if os.path.exists(classification_file):
        with open(classification_file, "r", encoding="utf-8") as f:
            print(f.read())

        # Step 8: Train/Validation/Test
    train_val_test_results_directory = os.path.join(args.results_dir, "train_val_test")
    scores = run_train_val_test(
        feature_names, feature_rows, target_values, train_val_test_results_directory,
        train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED,
        target_name=selected_target_column,
        use_gridsearch=use_gridsearch,
        ccp_alpha=ccp_alpha
    )
    print("\n=== TRAIN / VALIDATION / TEST ===")
    print(f"** Split: 70/15/15, seed={RANDOM_SEED} **\n")
    print(f"Validation accuracy = {scores['val_accuracy']:.4f}")
    print(f"Test accuracy = {scores['test_accuracy']:.4f} (results saved in {train_val_test_results_directory})\n")

    # Mostrar clasificación del modelo
    classification_file = os.path.join(train_val_test_results_directory, "model_classification.txt")
    if os.path.exists(classification_file):
        with open(classification_file, "r", encoding="utf-8") as f:
            print(f.read())

    # Comparación de accuracies
    plot_accuracy_comparison(
        showcase_acc=training_accuracy,
        validation_acc=test_accuracy,
        val_acc=scores["val_accuracy"],
        test_acc=scores["test_accuracy"],
        output_filename=os.path.join(args.results_dir, "accuracy_comparison.png")
    )

if __name__ == "__main__":
    main()