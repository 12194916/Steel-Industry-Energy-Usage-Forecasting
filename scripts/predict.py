"""
Energy Usage Prediction Script

This script loads the trained model and makes predictions on new data.

Usage:
    python predict.py --input data.csv --output predictions.csv
    python predict.py --input data.csv --output predictions.csv --model custom_model.pkl
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class EnergyUsagePredictor:
    """
    Energy usage predictor class
    """

    def __init__(self, model_path='../models/final_model.pkl',
                 scaler_path='../models/scaler.pkl',
                 model_info_path='../models/model_info.json'):
        """
        Initialize predictor with model and scaler

        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            model_info_path: Path to model metadata
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model_info_path = Path(model_info_path)

        # Load model
        print(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        # Load scaler if exists
        if self.scaler_path.exists():
            print(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
        else:
            print("Warning: Scaler not found. Predictions may be inaccurate for models requiring scaling.")
            self.scaler = None

        # Load model info
        if self.model_info_path.exists():
            with open(self.model_info_path, 'r') as f:
                self.model_info = json.load(f)
            print(f"Model type: {self.model_info.get('model_name', 'Unknown')}")
            print(f"Model R² score: {self.model_info.get('r2_score', 'Unknown'):.4f}")
        else:
            self.model_info = {}

    def preprocess_data(self, df):
        """
        Preprocess input data to match training format

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe ready for prediction
        """
        print("\nPreprocessing data...")

        # Make a copy
        df_processed = df.copy()

        # Convert numeric columns to proper types (fix for CSV string values)
        numeric_cols = [
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor',
            'NSM'
        ]
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Convert date to datetime if present
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], format='%d/%m/%Y %H:%M', dayfirst=True)

            # Extract temporal features
            df_processed['hour'] = df_processed['date'].dt.hour
            df_processed['day'] = df_processed['date'].dt.day
            df_processed['month'] = df_processed['date'].dt.month
            df_processed['dayofweek'] = df_processed['date'].dt.dayofweek
            df_processed['quarter'] = df_processed['date'].dt.quarter
            df_processed['dayofyear'] = df_processed['date'].dt.dayofyear
            df_processed['weekofyear'] = df_processed['date'].dt.isocalendar().week

            # Season
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'

            df_processed['season'] = df_processed['month'].apply(get_season)

            # Is weekend
            df_processed['is_weekend'] = (df_processed['dayofweek'] >= 5).astype(int)

            # Time of day
            def get_time_of_day(hour):
                if 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 22:
                    return 'Evening'
                else:
                    return 'Night'

            df_processed['time_of_day'] = df_processed['hour'].apply(get_time_of_day)

            # Cyclical encoding
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
            df_processed['dayofweek_sin'] = np.sin(2 * np.pi * df_processed['dayofweek'] / 7)
            df_processed['dayofweek_cos'] = np.cos(2 * np.pi * df_processed['dayofweek'] / 7)
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            df_processed['dayofyear_sin'] = np.sin(2 * np.pi * df_processed['dayofyear'] / 365)
            df_processed['dayofyear_cos'] = np.cos(2 * np.pi * df_processed['dayofyear'] / 365)

        # Create interaction features
        if 'Lagging_Current_Reactive.Power_kVarh' in df_processed.columns and 'Leading_Current_Reactive_Power_kVarh' in df_processed.columns:
            df_processed['total_reactive_power'] = (df_processed['Lagging_Current_Reactive.Power_kVarh'] +
                                                    df_processed['Leading_Current_Reactive_Power_kVarh'])
            df_processed['reactive_power_ratio'] = df_processed['Lagging_Current_Reactive.Power_kVarh'] / (df_processed['Leading_Current_Reactive_Power_kVarh'] + 1e-5)

        if 'Lagging_Current_Power_Factor' in df_processed.columns and 'Leading_Current_Power_Factor' in df_processed.columns:
            df_processed['power_factor_diff'] = (df_processed['Lagging_Current_Power_Factor'] -
                                                 df_processed['Leading_Current_Power_Factor'])
            df_processed['avg_power_factor'] = (df_processed['Lagging_Current_Power_Factor'] +
                                                df_processed['Leading_Current_Power_Factor']) / 2

        # Encode categoricals
        if 'Load_Type' in df_processed.columns:
            df_processed['Load_Type_encoded'] = df_processed['Load_Type'].map({
                'Light_Load': 0, 'Medium_Load': 1, 'Maximum_Load': 2
            }).fillna(0)

        if 'WeekStatus' in df_processed.columns:
            df_processed['WeekStatus_encoded'] = df_processed['WeekStatus'].map({
                'Weekday': 0, 'Weekend': 1
            }).fillna(0)

        if 'Day_of_week' in df_processed.columns:
            days_map = {day: idx for idx, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}
            df_processed['Day_of_week_encoded'] = df_processed['Day_of_week'].map(days_map).fillna(0)

        if 'season' in df_processed.columns:
            # Match LabelEncoder alphabetical order: Fall, Spring, Summer, Winter
            season_map = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
            df_processed['season_encoded'] = df_processed['season'].map(season_map).fillna(0)

        if 'time_of_day' in df_processed.columns:
            # Match LabelEncoder alphabetical order: Afternoon, Evening, Morning, Night
            time_map = {'Afternoon': 0, 'Evening': 1, 'Morning': 2, 'Night': 3}
            df_processed['time_of_day_encoded'] = df_processed['time_of_day'].map(time_map).fillna(0)

        # One-hot encoding
        if 'Load_Type' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['Load_Type'], prefix='LoadType')

        if 'WeekStatus' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['WeekStatus'], prefix='Week')

        if 'season' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['season'], prefix='Season')

        if 'time_of_day' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['time_of_day'], prefix='TimeOfDay')

        # Ensure all expected one-hot encoded columns exist
        expected_onehot_columns = [
            'LoadType_Light_Load', 'LoadType_Maximum_Load', 'LoadType_Medium_Load',
            'Week_Weekday', 'Week_Weekend',
            'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter',
            'TimeOfDay_Afternoon', 'TimeOfDay_Evening', 'TimeOfDay_Morning', 'TimeOfDay_Night'
        ]

        for col in expected_onehot_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Fill NaN values (use bfill() instead of deprecated method parameter)
        df_processed = df_processed.bfill().fillna(0)

        print(f"Preprocessed data shape: {df_processed.shape}")
        return df_processed

    def predict(self, df, return_features=False):
        """
        Make predictions on input data

        Args:
            df: Input dataframe
            return_features: Whether to return the feature matrix used for prediction

        Returns:
            Predictions array or tuple of (predictions, features)
        """
        # Preprocess data
        df_processed = self.preprocess_data(df)

        # Get feature columns from model info
        if 'feature_names' in self.model_info:
            expected_features = self.model_info['feature_names']

            # Ensure all expected features are present
            missing_features = set(expected_features) - set(df_processed.columns)
            if missing_features:
                print(f"\nWarning: Missing features: {missing_features}")
                print("Adding missing features with zero values...")
                for feat in missing_features:
                    df_processed[feat] = 0

            # Select only expected features in correct order
            X = df_processed[expected_features]
        else:
            # Exclude known non-feature columns
            exclude_cols = ['Usage_kWh', 'date', 'Day_of_week']
            feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
            X = df_processed[feature_cols]

        print(f"Feature matrix shape: {X.shape}")

        # Make predictions
        print("\nMaking predictions...")
        predictions = self.model.predict(X)

        print(f"Generated {len(predictions)} predictions")
        print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

        if return_features:
            return predictions, X
        return predictions


def main():
    """
    Main function to run predictions from command line
    """
    parser = argparse.ArgumentParser(description='Predict energy usage from input data')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output CSV file for predictions')
    parser.add_argument('--model', '-m', type=str, default='../models/final_model.pkl',
                       help='Path to model file (default: ../models/final_model.pkl)')
    parser.add_argument('--scaler', '-s', type=str, default='../models/scaler.pkl',
                       help='Path to scaler file (default: ../models/scaler.pkl)')

    args = parser.parse_args()

    # Load input data
    print(f"Loading input data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize predictor
    predictor = EnergyUsagePredictor(
        model_path=args.model,
        scaler_path=args.scaler
    )

    # Make predictions
    predictions = predictor.predict(df)

    # Create output dataframe
    df_output = df.copy()
    df_output['Predicted_Usage_kWh'] = predictions

    # If actual usage exists, calculate error metrics
    if 'Usage_kWh' in df_output.columns:
        df_output['Prediction_Error'] = df_output['Usage_kWh'] - df_output['Predicted_Usage_kWh']
        df_output['Absolute_Error'] = np.abs(df_output['Prediction_Error'])
        df_output['Percentage_Error'] = (df_output['Absolute_Error'] / df_output['Usage_kWh']) * 100

        print("\nPrediction Statistics:")
        print(f"Mean Absolute Error: {df_output['Absolute_Error'].mean():.4f}")
        print(f"Mean Percentage Error: {df_output['Percentage_Error'].mean():.2f}%")
        print(f"RMSE: {np.sqrt((df_output['Prediction_Error']**2).mean()):.4f}")

    # Save predictions
    df_output.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
