"""
Model extractor - extract frequent product models from CSV.
"""

import csv
import re
from collections import Counter
from pathlib import Path
from typing import List


def extract_frequent_models(csv_path: str, min_frequency: int = 3) -> List[str]:
    """
    Extract product models that appear frequently in the CSV.

    Args:
        csv_path: Path to the CSV file
        min_frequency: Minimum number of occurrences to be considered frequent

    Returns:
        List of frequent product model strings, sorted by frequency (descending)
    """
    path = Path(csv_path)
    models = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            content = row.get("content", "")
            table_data = row.get("table_data", "")
            combined_text = f"{content} {table_data}"

            # Pattern matches common product model formats:
            # - 2+ uppercase letters followed by 2+ digits and optional alphanumeric/dash
            # Examples: MHSR18ON8-S1, RUXYQ22CA, R32
            matches = re.findall(r'\b[A-Z]{2,}[0-9]{2,}[A-Z0-9\-]*\b', combined_text)
            models.extend(matches)

    counter = Counter(models)
    frequent_models = [
        model for model, count in counter.most_common()
        if count >= min_frequency
    ]

    return frequent_models
