from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from loan_approval_prediction.config import ProjectConfig
from loan_approval_prediction import predict
from pathlib import Path

app = FastAPI()
logger.info("Loan Approval Prediction Service is starting up.")
config = ProjectConfig()

model = predict.load_model(Path(config.model_dir) / config.model_file)


class LoanApprovalRequest(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    years_employed: int
    points: float
    # Add all necessary features here


@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval Prediction Service!"}


@app.post("/predict")
def predict_loan_approval(data: LoanApprovalRequest):
    logger.info(f"Received data for prediction: {data.model_dump()}")
    proba_prediction = predict.predict_proba(model, data.model_dump())
    result = {"loan_approval_probability": proba_prediction[0, 1]}
    logger.info(f"Prediction result: {result}")
    return result
