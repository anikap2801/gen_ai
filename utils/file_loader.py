import json
import os

import json
import os

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file at {file_path} was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error: Failed to decode JSON from {file_path}.")

def load_text(file_path):
    """
    Reads the content of a text file.
    Used for loading study material and logs.
    """
    if not os.path.exists(file_path):
        return ""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_json(data, file_path):
    """
    Saves a dictionary or list as a JSON file.
    Used for saving final results in the outputs/ folder.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)