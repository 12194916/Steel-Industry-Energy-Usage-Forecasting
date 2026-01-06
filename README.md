# ML Predictive Model - Steel Industry Energy Usage Forecasting

A comprehensive machine learning pipeline for predicting energy consumption (Usage_kWh) in the steel industry using temporal patterns, power metrics, and advanced feature engineering techniques.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models & Algorithms](#models--algorithms)
- [Usage Guide](#usage-guide)
- [Results & Performance](#results--performance)
- [Technical Details](#technical-details)

---

## Project Overview

This project implements a complete end-to-end machine learning solution for energy usage prediction with:
- **79 engineered features** from 11 original features
- **9 different ML algorithms** benchmarked
- **Hyperparameter optimization** using RandomizedSearchCV
- **Production-ready prediction script** with full preprocessing pipeline
- **Comprehensive EDA** with visualizations

**Target Variable**: `Usage_kWh` (Energy consumption in kilowatt-hours)

**Use Case**: Predict future energy consumption based on historical patterns, power metrics, and temporal factors to optimize energy management in steel manufacturing.

---

## Project Structure

```
ML-Predictive-model/
├── dataset/
│   └── Steel_industry_data.csv          # Raw dataset (35,041 records)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Data cleaning & validation
│   ├── 02_exploratory_data_analysis.ipynb  # Comprehensive EDA
│   ├── 03_feature_engineering.ipynb     # Feature creation (79 features)
│   ├── 04_model_training.ipynb          # Train 9 ML models
│   ├── 05_hyperparameter_tuning.ipynb   # Optimize top 3 models
│   └── 06_using_the_model.ipynb         # Deployment & usage examples
├── models/
│   ├── final_model.pkl                  # Best trained model
│   ├── scaler.pkl                       # Feature scaler
│   ├── model_info.json                  # Model metadata
│   ├── model_comparison.csv             # Benchmark results
│   └── feature_importance.csv           # Feature importance scores
├── processed_data/
│   ├── steel_data_cleaned.csv           # Preprocessed data
│   ├── steel_data_featured.csv          # Engineered features
│   └── feature_names.txt                # List of all features
├── sample_inputs/                       # Sample CSV files for testing
│   ├── 01_weekday_morning_light_load.csv
│   ├── 02_weekday_afternoon_medium_load.csv
│   ├── 03_weekday_evening_maximum_load.csv
│   ├── 04_weekend_saturday_light_load.csv
│   ├── 05_weekend_sunday_medium_load.csv
│   ├── 06_single_prediction_example.csv
│   └── README.md                        # Sample files documentation
├── scripts/
│   └── predict.py                       # Production prediction script
├── .gitignore                           # Git ignore file (excludes .pkl)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Dataset Information

### Raw Dataset
- **Source**: Steel industry power consumption data
- **Records**: 35,041 observations
- **Time Period**: January 2018 - December 2018
- **Frequency**: 15-minute intervals (96 readings per day)
- **Format**: CSV with datetime index

### Original Features (11 columns)
| Feature | Description | Type |
|---------|-------------|------|
| `date` | Timestamp (DD/MM/YYYY HH:MM) | Datetime |
| `Usage_kWh` | Energy consumption (target) | Float |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power | Float |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power | Float |
| `CO2(tCO2)` | CO2 emissions in tons | Float |
| `Lagging_Current_Power_Factor` | Lagging power factor | Float |
| `Leading_Current_Power_Factor` | Leading power factor | Float |
| `NSM` | Number of seconds from midnight | Integer |
| `WeekStatus` | Weekday or Weekend | Categorical |
| `Day_of_week` | Monday - Sunday | Categorical |
| `Load_Type` | Light/Medium/Maximum Load | Categorical |

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd ML-Predictive-model
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (1.3+)
- **Gradient Boosting**: xgboost (2.0+), lightgbm (4.1+)
- **Deep Learning**: tensorflow (2.15+)
- **Notebook**: jupyterlab, ipykernel

**Important**: The project requires **NumPy < 2.0** for TensorFlow compatibility:
```bash
pip install "numpy<2.0"
```

---

## Data Processing Pipeline

### 1. Data Preprocessing ([01_data_preprocessing.ipynb](notebooks/01_data_preprocessing.ipynb))

**Operations**:
- ✅ Load raw CSV data
- ✅ Convert `date` column to datetime format
- ✅ Sort data chronologically
- ✅ Check for missing values (Result: 0 missing)
- ✅ Check for duplicates (Result: 0 duplicates)
- ✅ Validate data types
- ✅ Detect outliers using IQR method
- ✅ Statistical summary and validation

**Output**: `steel_data_cleaned.csv` (35,040 rows × 11 columns)

### 2. Exploratory Data Analysis ([02_exploratory_data_analysis.ipynb](notebooks/02_exploratory_data_analysis.ipynb))

**Analysis Performed**:
- 📊 Target variable distribution (Usage_kWh)
- 📈 Time series visualization
- 🕒 Hourly/daily/monthly patterns
- 📉 Correlation analysis (heatmap)
- 🔍 Load type impact analysis
- 🌡️ Weekday vs. weekend patterns
- ⚡ Power factor distributions
- 🌍 CO2 emissions relationship

**Key Insights**:
- Strong hourly patterns with peaks during working hours
- Significant difference between weekday and weekend consumption
- High correlation between reactive power and usage
- Seasonal variations in energy consumption

---

## Feature Engineering

### Complete Feature Engineering Process ([03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb))

**From 11 → 79 features**

### A. Temporal Features (18 features)

#### Basic Time Features
```python
- hour (0-23)
- day (1-31)
- month (1-12)
- dayofweek (0-6, Monday=0)
- quarter (1-4)
- dayofyear (1-365)
- weekofyear (1-52)
- is_weekend (0/1)
```

#### Derived Temporal
```python
- season (Winter/Spring/Summer/Fall)
- time_of_day (Morning/Afternoon/Evening/Night)
```

#### Cyclical Encoding (8 features)
Captures circular nature of time:
```python
- hour_sin, hour_cos (24-hour cycle)
- dayofweek_sin, dayofweek_cos (7-day cycle)
- month_sin, month_cos (12-month cycle)
- dayofyear_sin, dayofyear_cos (365-day cycle)
```

**Why?** Models can understand that hour 23 and hour 0 are close together.

### B. Lag Features (16 features)

Capture recent historical patterns:
```python
Lag periods: [1, 2, 3, 4, 8, 12, 24, 96]
# 1=15min, 4=1hr, 8=2hr, 24=6hr, 96=24hr ago

For each period:
- usage_lag_{period}
- lagging_power_lag_{period}
```

**Example**: `usage_lag_4` = energy usage from 1 hour ago

### C. Rolling Window Features (16 features)

Statistical summaries over time windows:
```python
Windows: [4, 8, 12, 24] (1hr, 2hr, 3hr, 6hr)

For each window:
- usage_rolling_mean_{window}
- usage_rolling_std_{window}
- usage_rolling_min_{window}
- usage_rolling_max_{window}
```

**Purpose**: Capture trends, volatility, and ranges

### D. Interaction Features (4 features)

Domain-specific engineered features:
```python
1. total_reactive_power = lagging + leading reactive power
2. power_factor_diff = lagging - leading power factor
3. avg_power_factor = (lagging + leading) / 2
4. reactive_power_ratio = lagging / (leading + epsilon)
```

### E. Categorical Encoding

#### Label Encoding (5 features)
Ordinal encoding using alphabetical order (matching scikit-learn LabelEncoder):
```python
- Load_Type_encoded: {Light_Load: 0, Maximum_Load: 1, Medium_Load: 2}
- WeekStatus_encoded: {Weekday: 0, Weekend: 1}
- Day_of_week_encoded: {Monday: 0, ..., Sunday: 6}
- season_encoded: {Fall: 0, Spring: 1, Summer: 2, Winter: 3}
- time_of_day_encoded: {Afternoon: 0, Evening: 1, Morning: 2, Night: 3}
```

#### One-Hot Encoding (13 features)
For tree-based models:
```python
Load_Type → LoadType_Light_Load, LoadType_Maximum_Load, LoadType_Medium_Load
WeekStatus → Week_Weekday, Week_Weekend
season → Season_Fall, Season_Spring, Season_Summer, Season_Winter
time_of_day → TimeOfDay_Afternoon, TimeOfDay_Evening, TimeOfDay_Morning, TimeOfDay_Night
```

### F. Missing Value Handling
```python
# Lag and rolling features have NaN at the start
# Strategy: Backward fill then fill with 0
df[lag_columns] = df[lag_columns].bfill().fillna(0)
```

**Final Output**: `steel_data_featured.csv` (35,040 rows × **79 features**)

---

## Models & Algorithms

### Model Training ([04_model_training.ipynb](notebooks/04_model_training.ipynb))

**9 Models Benchmarked**:

#### 1. **Linear Regression** (Baseline)
```python
- Simple linear model
- Uses scaled features
- Fast training, interpretable
```

#### 2. **Ridge Regression**
```python
- L2 regularization (alpha=1.0)
- Prevents overfitting
- Handles multicollinearity
```

#### 3. **Lasso Regression**
```python
- L1 regularization (alpha=0.1)
- Feature selection capability
- Sparse coefficients
```

#### 4. **Decision Tree Regressor**
```python
- Max depth: 15
- Min samples split: 10
- No feature scaling needed
```

#### 5. **Random Forest Regressor**
```python
- 100 estimators
- Max depth: 20
- Min samples split: 10
- Parallel processing (n_jobs=-1)
```

#### 6. **Gradient Boosting Regressor**
```python
- 100 estimators
- Learning rate: 0.1
- Max depth: 7
- Sequential boosting
```

#### 7. **XGBoost Regressor**
```python
- 100 estimators
- Learning rate: 0.1
- Max depth: 7
- Subsample: 0.8
- Optimized gradient boosting
```

#### 8. **LightGBM Regressor**
```python
- 100 estimators
- Learning rate: 0.1
- Max depth: 7
- Leaf-wise tree growth
- Fast training on large datasets
```

#### 9. **Neural Network (TensorFlow/Keras)**
```python
Architecture:
- Dense(128, relu) + Dropout(0.3)
- Dense(64, relu) + Dropout(0.2)
- Dense(32, relu) + Dropout(0.2)
- Dense(16, relu)
- Dense(1) - Output

Optimizer: Adam (lr=0.001)
Loss: MSE
Early stopping: patience=10
Batch size: 64
Max epochs: 100
```

### Data Splitting Strategy

**Time-series aware split** (chronological, not random):
```python
Training: First 80% (28,032 samples)
Testing: Last 20% (7,008 samples)
```

**Why chronological?** Prevents data leakage from future to past in time series.

### Evaluation Metrics

All models evaluated using:
- **RMSE** (Root Mean Squared Error) - Lower is better
- **MAE** (Mean Absolute Error) - Lower is better
- **R² Score** (Coefficient of Determination) - Higher is better (max 1.0)
- **MAPE** (Mean Absolute Percentage Error) - Lower is better

### Hyperparameter Tuning ([05_hyperparameter_tuning.ipynb](notebooks/05_hyperparameter_tuning.ipynb))

**Top 3 models optimized** using RandomizedSearchCV:

#### XGBoost Parameter Grid
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}
```

#### LightGBM Parameter Grid
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 20, 30]
}
```

#### Random Forest Parameter Grid
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

**Tuning Strategy**:
- RandomizedSearchCV with 20 iterations (XGBoost, LightGBM)
- RandomizedSearchCV with 15 iterations (Random Forest)
- 3-fold cross-validation
- Scoring metric: R²

**Output**: `final_model.pkl` - Best performing optimized model

---

## Usage Guide

### Running the Notebooks

**Execute in order**:

```bash
# 1. Start JupyterLab
jupyter lab

# 2. Run notebooks sequentially:
01_data_preprocessing.ipynb       # ~2 min
02_exploratory_data_analysis.ipynb # ~5 min
03_feature_engineering.ipynb      # ~3 min
04_model_training.ipynb           # ~15-30 min (NN training)
05_hyperparameter_tuning.ipynb    # ~20-40 min (RandomizedSearch)
06_using_the_model.ipynb          # ~2 min
```

### Making Predictions

#### Option 1: Python Script (Command Line)

```bash
# Basic prediction
python scripts/predict.py --input data.csv --output predictions.csv

# With custom model
python scripts/predict.py \
  --input new_data.csv \
  --output results.csv \
  --model models/xgboost_tuned.pkl
```

**Input CSV Requirements**:
Must contain these columns:
- `date` (format: DD/MM/YYYY HH:MM)
- `Lagging_Current_Reactive.Power_kVarh`
- `Leading_Current_Reactive_Power_kVarh`
- `CO2(tCO2)`
- `Lagging_Current_Power_Factor`
- `Leading_Current_Power_Factor`
- `NSM`
- `WeekStatus` (Weekday/Weekend)
- `Day_of_week` (Monday-Sunday)
- `Load_Type` (Light_Load/Medium_Load/Maximum_Load)

#### Option 2: Python API

```python
from scripts.predict import EnergyUsagePredictor
import pandas as pd

# Load model
predictor = EnergyUsagePredictor(
    model_path='models/final_model.pkl',
    scaler_path='models/scaler.pkl'
)

# Load data
df = pd.read_csv('new_data.csv')

# Make predictions
predictions = predictor.predict(df)

# Add to dataframe
df['Predicted_Usage_kWh'] = predictions
df.to_csv('results.csv', index=False)
```

#### Option 3: Single Prediction

```python
# Create single input
single_input = pd.DataFrame([{
    'date': '01/01/2018 12:00',
    'Lagging_Current_Reactive.Power_kVarh': 5.5,
    'Leading_Current_Reactive_Power_kVarh': 0.0,
    'CO2(tCO2)': 0.01,
    'Lagging_Current_Power_Factor': 65.0,
    'Leading_Current_Power_Factor': 100.0,
    'NSM': 43200,
    'WeekStatus': 'Weekday',
    'Day_of_week': 'Monday',
    'Load_Type': 'Medium_Load'
}])

# Predict
prediction = predictor.predict(single_input)
print(f"Predicted Usage: {prediction[0]:.2f} kWh")
```

---

## Results & Performance

### Model Comparison (Before Tuning)

Typical performance ranking:
1. **XGBoost / LightGBM** - Best R² (0.95-0.98)
2. Random Forest - Very good R² (0.93-0.96)
3. Gradient Boosting - Good R² (0.92-0.95)
4. Neural Network - Good R² (0.90-0.94)
5. Ridge / Lasso - Moderate R² (0.75-0.85)
6. Linear Regression - Baseline R² (0.70-0.80)
7. Decision Tree - Prone to overfitting

### Final Model Performance (After Tuning)

Best model metrics saved in `models/model_info.json`:
- Model name, type, and parameters
- R², RMSE, MAE, MAPE scores
- Training/test set sizes
- Feature count and names

### Feature Importance

Top features typically include:
1. Lag features (usage_lag_1, usage_lag_4)
2. Rolling means (usage_rolling_mean_24)
3. Power metrics (Lagging_Current_Reactive.Power_kVarh)
4. Hour of day (hour, hour_sin, hour_cos)
5. Load type encodings

Saved in: `models/feature_importance.csv`

---

## Technical Details

### Preprocessing Pipeline in predict.py

The prediction script automatically applies:
1. ✅ Datetime parsing and temporal feature extraction
2. ✅ Cyclical encoding (sin/cos transformations)
3. ✅ Lag feature creation (if sufficient data)
4. ✅ Rolling window statistics
5. ✅ Interaction features
6. ✅ Label encoding (alphabetical order matching)
7. ✅ One-hot encoding with missing column handling
8. ✅ Missing value imputation (backward fill + zero fill)
9. ✅ Feature alignment to training schema

### Key Design Decisions

#### Why chronological train/test split?
- Prevents data leakage in time series
- Tests model's ability to predict future data
- More realistic evaluation

#### Why 79 features from 11?
- Rich temporal patterns (hourly, daily, seasonal)
- Historical context (lag features)
- Trend indicators (rolling statistics)
- Domain knowledge (power factor interactions)

#### Why multiple encoding strategies?
- Label encoding: For ordinal relationships
- One-hot encoding: For tree-based models (no ordinality assumption)
- Cyclical encoding: For circular time features

#### Why tree-based models perform best?
- Handle non-linear relationships
- Automatic feature interactions
- Robust to outliers
- No feature scaling required
- Capture complex temporal patterns

---

## Troubleshooting

### Common Issues

**Issue 1: NumPy compatibility error**
```
AttributeError: `np.complex_` was removed in NumPy 2.0
```
**Solution**:
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
```

**Issue 2: Feature count mismatch**
```
ValueError: X has 71 features, but model expects 76
```
**Solution**: Re-run notebooks 03-05 to regenerate features consistently

**Issue 3: Missing columns in prediction**
```
KeyError: 'LoadType_Maximum_Load'
```
**Solution**: Updated predict.py ensures all one-hot columns exist (already fixed)

---

## File Outputs

### Models Directory
- `final_model.pkl` - Best performing model
- `xgboost_tuned.pkl` - Tuned XGBoost
- `lightgbm_tuned.pkl` - Tuned LightGBM
- `random_forest_tuned.pkl` - Tuned Random Forest
- `scaler.pkl` - StandardScaler for features
- `model_info.json` - Model metadata
- `model_comparison.csv` - Benchmark results
- `feature_importance.csv` - Feature rankings

### Processed Data Directory
- `steel_data_cleaned.csv` - Cleaned dataset (11 features)
- `steel_data_featured.csv` - Engineered features (79 features)
- `feature_names.txt` - Complete feature list

---

## Contributing

To extend this project:
1. Add new features in `03_feature_engineering.ipynb`
2. Update `predict.py` to match new features
3. Add new models in `04_model_training.ipynb`
4. Re-run tuning in `05_hyperparameter_tuning.ipynb`

---

## References

**Libraries Used**:
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- TensorFlow: https://www.tensorflow.org/

**Dataset**: Steel industry energy consumption data (2018)

---

## License

This project is for educational and research purposes.

---

## Contact

For questions or issues, please create an issue in the project repository.

---

**Last Updated**: January 2026
**Python Version**: 3.8+
**Key Dependencies**: pandas, scikit-learn, xgboost, lightgbm, tensorflow
