"""
Main entry point for the Flood Risk Prediction system.
This script orchestrates the entire ML pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

from . import config
from .data_preprocessing import preprocess_data
from .train import train_and_save
from .evaluate import evaluate_model, save_evaluation_report
from .predict import load_model, load_scaler, batch_predict, classify_prediction


def main():
    """
    Main pipeline function that orchestrates the entire ML workflow:
    1. Data preprocessing (load, clean, split, scale)
    2. Model training
    3. Model evaluation
    4. Save results
    """
    
    print("\n" + "="*70)
    print("FLOOD RISK PREDICTION - MACHINE LEARNING PIPELINE")
    print("="*70)
    
    try:
        # ==========================================
        # 1. DATA PREPROCESSING
        # ==========================================
        X_train, X_test, y_train, y_test = preprocess_data(config.RAW_DATA_PATH)
        
        # ==========================================
        # 2. MODEL TRAINING
        # ==========================================
        model, feature_importance = train_and_save(X_train, y_train, config.MODEL_PATH)
        
        # ==========================================
        # 3. MODEL EVALUATION
        # ==========================================
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # ==========================================
        # 4. SAVE EVALUATION REPORT
        # ==========================================
        save_evaluation_report(results, "reports/evaluation_report.txt")
        
        # Save feature importance
        feature_importance.to_csv("reports/feature_importance.csv", index=False)
        print(f"✓ Feature importance saved to: reports/feature_importance.csv")
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nModel is ready for deployment!")
        print(f"  - Model saved: {config.MODEL_PATH}")
        print(f"  - Scaler saved: {config.SCALER_PATH}")
        print(f"  - Evaluation report: reports/evaluation_report.txt")
        print("="*70 + "\n")
        
        return model, results
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure the raw data CSV file exists at:")
        print(f"  {config.RAW_DATA_PATH}")
        print("\nExpected CSV columns:")
        print(f"  {config.FEATURE_COLUMNS + [config.TARGET_COLUMN]}")
        return None, None
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None, None


def predict_on_new_data(input_csv_path: str):
    """
    Load a trained model and make predictions on new data.
    
    Args:
        input_csv_path (str): Path to CSV file with new data
    """
    print("\n" + "="*70)
    print("PREDICTION ON NEW DATA")
    print("="*70)
    
    try:
        # Load model and scaler
        model = load_model(config.MODEL_PATH)
        scaler = load_scaler(config.SCALER_PATH)
        
        # Make predictions
        predictions_df = batch_predict(
            model,
            input_csv_path,
            scaler,
            config.NUMERICAL_FEATURES
        )
        
        # Save results
        output_path = input_csv_path.replace(".csv", "_predictions.csv")
        predictions_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")
        
        # Display sample results
        print("\nSample Predictions:")
        print(predictions_df[["prediction", "flood_probability"]].head(10))
        
        print("="*70 + "\n")
        return predictions_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


if __name__ == "__main__":
    # Run the main pipeline
    main()
