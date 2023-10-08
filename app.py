
from fastapi import FastAPI, HTTPException
import xgboost as xgb
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
app = FastAPI()

# Load your trained XGBoost model
model = xgb.Booster(model_file="/Users/hrishikeshkesavannair/Desktop/flask/my_model_xgboost_1_4_2.model")

@app.post("/predict/")
async def predict(data: dict):
    try:
        # Preprocess input data if needed (e.g., convert to a format expected by your model)
        input_data = np.array([list(data.values())], dtype=np.float32)

        # Make predictions using your loaded XGBoost model
        dmatrix = xgb.DMatrix(input_data)
        predictions = model.predict(dmatrix)

        # Return the predictions as a JSON response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Instrument the FastAPI app for Prometheus
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8060)

