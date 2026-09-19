"""
The Pareidolia Paradox: Training Entry Point
Runs 5-Fold Stratified Cross-Validation on Lunar Terrain crops.
"""
import sys
import os
from train_kfold import train_kfold

if __name__ == "__main__":
    train_kfold()
