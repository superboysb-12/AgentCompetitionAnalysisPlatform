"""
Relation extractor package.
"""

from .relation_extractor import RelationExtractor
from .csv_processor import load_pages_from_csv

__all__ = ["RelationExtractor", "load_pages_from_csv"]
