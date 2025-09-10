
import csv
import os
import shutil

def ensure_result_directories(base_results_path: str):
    os.makedirs(os.path.join(base_results_path, "showcase"), exist_ok=True)
    os.makedirs(os.path.join(base_results_path, "validation"), exist_ok=True)

def clear_directory_contents(directory_path: str):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return
    for item_name in os.listdir(directory_path):
        if item_name.lower() == "readme.md":
            continue
        item_path = os.path.join(directory_path, item_name)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def reset_result_directories(base_results_path: str):
    ensure_result_directories(base_results_path)
    clear_directory_contents(os.path.join(base_results_path, "showcase"))
    clear_directory_contents(os.path.join(base_results_path, "validation"))

def save_text_to_file(file_path: str, text_content: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_content if text_content is not None else "")

def load_csv_data(csv_file_path: str):
    with open(csv_file_path, "r", newline="") as file:
        rows = list(csv.reader(file))
    header = rows[0]
    data_rows = rows[1:]
    features = [row[:-1] for row in data_rows]
    targets = [row[-1] for row in data_rows]
    return header[:-1], features, targets
