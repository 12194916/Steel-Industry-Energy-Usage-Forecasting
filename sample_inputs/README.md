# Sample Input Files for Energy Usage Prediction App

This folder contains 6 sample CSV files that can be used to test the prediction model. These files contain **only the original features** (11 columns) and are ready to be processed by the app.

---

## File Descriptions

### 1. `01_weekday_morning_light_load.csv`
- **Scenario**: Weekday morning shift (6:00 AM - 8:15 AM)
- **Load Type**: Light Load
- **Day**: Friday
- **Rows**: 10 records (15-minute intervals)
- **Use Case**: Testing low energy consumption during early work hours

---

### 2. `02_weekday_afternoon_medium_load.csv`
- **Scenario**: Weekday lunch/afternoon (12:00 PM - 2:15 PM)
- **Load Type**: Medium Load
- **Day**: Monday
- **Rows**: 10 records
- **Use Case**: Testing moderate energy consumption during peak production hours

---

### 3. `03_weekday_evening_maximum_load.csv`
- **Scenario**: Weekday evening (6:00 PM - 8:15 PM)
- **Load Type**: Maximum Load
- **Day**: Thursday
- **Rows**: 10 records
- **Use Case**: Testing high energy consumption during maximum production

---

### 4. `04_weekend_saturday_light_load.csv`
- **Scenario**: Weekend morning (10:00 AM - 12:15 PM)
- **Load Type**: Light Load
- **Day**: Saturday
- **Rows**: 10 records
- **Use Case**: Testing weekend patterns with minimal operations

---

### 5. `05_weekend_sunday_medium_load.csv`
- **Scenario**: Weekend afternoon (2:00 PM - 4:15 PM)
- **Load Type**: Medium Load
- **Day**: Sunday
- **Rows**: 10 records
- **Use Case**: Testing Sunday operations with moderate load

---

### 6. `06_single_prediction_example.csv`
- **Scenario**: Single time point prediction
- **Load Type**: Light Load
- **Day**: Wednesday, 9:30 AM
- **Rows**: 1 record
- **Use Case**: Testing single prediction (real-time use case)

---

## Input File Format

All files contain these **11 columns** (original features only):

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `date` | String | DD/MM/YYYY HH:MM format | 15/03/2024 06:00 |
| `Lagging_Current_Reactive.Power_kVarh` | Float | Lagging reactive power | 4.32 |
| `Leading_Current_Reactive_Power_kVarh` | Float | Leading reactive power | 0 |
| `CO2(tCO2)` | Float | CO2 emissions | 0 or 0.01 |
| `Lagging_Current_Power_Factor` | Float | Lagging power factor | 65.85 |
| `Leading_Current_Power_Factor` | Float | Leading power factor | 100 |
| `NSM` | Integer | Seconds from midnight | 21600 |
| `WeekStatus` | String | Weekday or Weekend | Weekday |
| `Day_of_week` | String | Day name | Friday |
| `Load_Type` | String | Light_Load, Medium_Load, or Maximum_Load | Light_Load |

---

## How to Use These Files

### Option 1: Command Line Script

```bash
# Predict using sample file
python scripts/predict.py --input sample_inputs/01_weekday_morning_light_load.csv --output results.csv

# Multiple files
python scripts/predict.py --input sample_inputs/03_weekday_evening_maximum_load.csv --output predictions_evening.csv
```

### Option 2: Python API

```python
from scripts.predict import EnergyUsagePredictor
import pandas as pd

# Load predictor
predictor = EnergyUsagePredictor()

# Load sample file
df = pd.read_csv('sample_inputs/02_weekday_afternoon_medium_load.csv')

# Make predictions
predictions = predictor.predict(df)

# View results
print(f"Predicted usage: {predictions}")
```

### Option 3: In Your App

```python
# The app will automatically:
# 1. Load the CSV file
# 2. Validate the 11 required columns
# 3. Apply feature engineering (11 → 79 features)
# 4. Make predictions using the trained model
# 5. Return predicted Usage_kWh values
```

---

## What Happens During Prediction?

When you upload these files to the app, the preprocessing pipeline automatically:

1. ✅ **Parses datetime** from the `date` column
2. ✅ **Extracts temporal features** (hour, day, month, season, etc.)
3. ✅ **Creates cyclical encodings** (sin/cos for time features)
4. ✅ **Generates lag features** (if sufficient historical data)
5. ✅ **Computes rolling statistics** (means, std, min, max)
6. ✅ **Creates interaction features** (power factor combinations)
7. ✅ **Encodes categoricals** (label encoding + one-hot encoding)
8. ✅ **Aligns features** to match the 79 features used during training
9. ✅ **Makes predictions** using the trained model

**Result**: You get `Predicted_Usage_kWh` for each row!

---

## Expected Output Format

After prediction, you'll get an output CSV with additional columns:

```csv
date,Lagging_Current_Reactive.Power_kVarh,...,Load_Type,Predicted_Usage_kWh
15/03/2024 06:00,4.32,...,Light_Load,3.85
15/03/2024 06:15,3.64,...,Light_Load,3.52
...
```

If the input file contains actual `Usage_kWh` values, you'll also get error metrics:
```csv
...,Predicted_Usage_kWh,Prediction_Error,Absolute_Error,Percentage_Error
...,3.85,-0.07,0.07,1.85
```

---

## Creating Your Own Input Files

To create your own test files:

1. **Use the same 11 columns** (exact names and order)
2. **Date format**: DD/MM/YYYY HH:MM (e.g., 25/12/2024 14:30)
3. **Categorical values**:
   - `WeekStatus`: "Weekday" or "Weekend"
   - `Day_of_week`: "Monday", "Tuesday", ..., "Sunday"
   - `Load_Type`: "Light_Load", "Medium_Load", or "Maximum_Load"
4. **NSM calculation**: `hour * 3600 + minute * 60`
   - Example: 14:30 = 14 * 3600 + 30 * 60 = 52200

---

## Testing Different Scenarios

Use these files to test:
- ✅ Different times of day (morning, afternoon, evening, night)
- ✅ Different days of week (weekday vs weekend)
- ✅ Different load types (light, medium, maximum)
- ✅ Single vs batch predictions
- ✅ Model performance across different conditions

---

## Notes

- **No preprocessing needed**: Just upload the CSV as-is
- **Minimum 1 row**: Can predict single data points
- **Recommended**: 100+ rows for lag features to work properly
- **File size**: Keep under 10MB for web apps
- **Encoding**: UTF-8 without BOM

---

**Ready to use in your app!** 🚀
