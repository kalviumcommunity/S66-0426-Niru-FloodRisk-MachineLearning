"""
Main entry point to run the Flood Risk Prediction ML pipeline.
This script can be run from the project root.
"""

import sys
from src.main import main, predict_on_new_data


if __name__ == "__main__":
    # Example 1: Run the complete pipeline
    # This will train the model, evaluate it, and save results
    model, results = main()
    
    # Example 2: Make predictions on new data (uncomment to use)
    # predictions = predict_on_new_data("path/to/new_data.csv")
