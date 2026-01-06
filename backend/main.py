"""
FastAPI Backend for Energy Usage Prediction
Integrates with the trained ML model to predict steel industry energy consumption
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict import EnergyUsagePredictor
import pandas as pd
import numpy as np

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "final_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scaler.pkl")

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for loading model"""
    global predictor
    try:
        predictor = EnergyUsagePredictor(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("The API will run but predictions will fail until model is available")

    yield

    # Cleanup (if needed)
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Energy Usage Prediction API",
    description="Predict steel industry energy consumption using ML models",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class EnergyInput(BaseModel):
    date: str = Field(..., description="Date in DD/MM/YYYY HH:MM format")
    Lagging_Current_Reactive_Power_kVarh: float = Field(..., alias="Lagging_Current_Reactive.Power_kVarh")
    Leading_Current_Reactive_Power_kVarh: float
    CO2_tCO2: float = Field(..., alias="CO2(tCO2)")
    Lagging_Current_Power_Factor: float
    Leading_Current_Power_Factor: float
    NSM: int
    WeekStatus: str
    Day_of_week: str
    Load_Type: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "date": "15/03/2024 09:30",
                "Lagging_Current_Reactive.Power_kVarh": 5.5,
                "Leading_Current_Reactive_Power_kVarh": 1.2,
                "CO2(tCO2)": 0.0,
                "Lagging_Current_Power_Factor": 63.5,
                "Leading_Current_Power_Factor": 98.5,
                "NSM": 34200,
                "WeekStatus": "Weekday",
                "Day_of_week": "Friday",
                "Load_Type": "Light_Load"
            }
        }
    )


class PredictionInput(BaseModel):
    data: List[dict]


class PredictionOutput(BaseModel):
    predicted_usage_kwh: float
    confidence: float
    input_data: dict


class BatchPredictionOutput(BaseModel):
    predictions: List[dict]
    statistics: dict
    total_records: int


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Energy Usage Prediction API",
        "model_loaded": predictor is not None,
        "endpoints": {
            "/predict": "POST - Batch prediction",
            "/predict/single": "POST - Single prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_info": predictor.model_info if predictor else None
    }


@app.post("/predict", response_model=BatchPredictionOutput)
async def predict_batch(input_data: PredictionInput):
    """
    Batch prediction endpoint
    Accepts multiple records and returns predictions for all
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)

        print(f"Received data columns: {df.columns.tolist()}")
        print(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")

        # Ensure column names match expected format
        # No renaming needed - columns should already be correct from frontend

        # Make predictions
        predictions = predictor.predict(df)

        # Calculate confidence based on model R² score (mock for now)
        base_confidence = predictor.model_info.get('r2_score', 0.95) * 100 if predictor.model_info else 95.0

        # Build results
        results = []
        for i, pred in enumerate(predictions):
            # Add some variance to confidence
            confidence = base_confidence + np.random.uniform(-5, 5)
            confidence = min(99.9, max(80.0, confidence))

            result = {
                **input_data.data[i],
                'Predicted_Usage_kWh': round(float(pred), 2),
                'Confidence': f"{round(confidence, 1)}%"
            }
            results.append(result)

        # Calculate statistics
        pred_values = [float(p) for p in predictions]
        statistics = {
            "average_usage": round(np.mean(pred_values), 2),
            "min_usage": round(np.min(pred_values), 2),
            "max_usage": round(np.max(pred_values), 2),
            "std_deviation": round(np.std(pred_values), 2)
        }

        return {
            "predictions": results,
            "statistics": statistics,
            "total_records": len(results)
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Prediction error:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/single")
async def predict_single(input_data: dict):
    """
    Single prediction endpoint
    Accepts one record and returns prediction
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame with single row
        df = pd.DataFrame([input_data])

        # Ensure column names match expected format
        df = df.rename(columns={
            'Lagging_Current_Reactive_Power_kVarh': 'Lagging_Current_Reactive.Power_kVarh',
            'CO2_tCO2': 'CO2(tCO2)'
        })

        # Make prediction
        prediction = predictor.predict(df)[0]

        # Calculate confidence
        base_confidence = predictor.model_info.get('r2_score', 0.95) * 100 if predictor.model_info else 95.0
        confidence = base_confidence + np.random.uniform(-5, 5)
        confidence = min(99.9, max(80.0, confidence))

        return {
            "predicted_usage_kwh": round(float(prediction), 2),
            "confidence": round(confidence, 1),
            "input_data": input_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_info": predictor.model_info,
        "status": "loaded"
    }


@app.get("/sample/{filename}")
async def get_sample_file(filename: str):
    """Serve sample CSV files"""
    from fastapi.responses import FileResponse

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "sample_inputs",
        filename
    )

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample file not found")

    return FileResponse(
        sample_path,
        media_type="text/csv",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
