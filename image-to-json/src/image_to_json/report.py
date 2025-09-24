import csv
from datetime import datetime
from pathlib import Path

from .paths import get_path_to_evals

def save_predictions_to_disk(predictions_data: list[dict]) -> str:
    """Saves predictions to a csv file with a timestamp and returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_name = f"predictions_{timestamp}.csv"
    csv_file_name = str(Path(get_path_to_evals()) / csv_file_name)

    # Write predictions to CSV
    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = ['ground_truth', 'predicted', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(predictions_data)

    return csv_file_name
