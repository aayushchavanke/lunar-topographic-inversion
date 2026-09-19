"""
The Pareidolia Paradox: Official Inference Entry Point
Runs 40-Pass Physics-Safe TTA (8 views x 5 folds) + Shape-from-Shading (SfS) + Gated Multi-Scale Center Zoom Refinement.
Generates the verified 2,000-row submission.csv.
"""
import sys
import os
from inference_ensemble_tta import run_tta_ensemble_inference

if __name__ == "__main__":
    run_tta_ensemble_inference()
