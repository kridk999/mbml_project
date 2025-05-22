"""
MBML - Machine Learning for CS:GO round prediction.
"""

__version__ = "0.1.0"

from mbml.dataset import CSGORoundDataset, collate_variable_length_rounds, create_dataloaders
from mbml.model import LSTMWinPredictor
