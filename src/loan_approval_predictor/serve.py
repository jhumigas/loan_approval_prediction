from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from loan_approval_predictor.config import ProjectConfig
from loan_approval_predictor import predict

app = FastAPI()
logger.info("Loan Approval Prediction Service is starting up.")
config = ProjectConfig.from_yaml("config.yml")

model = predict.load_model(config.model_path)


class LoanApprovalRequest(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    years_employed: int
    points: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval Prediction Service!"}


@app.post("/predict")
def predict_loan_approval(data: LoanApprovalRequest):
    logger.info(f"Received data for prediction: {data.model_dump()}")
    proba_prediction = predict.predict_proba(model, data.model_dump())
    result = {
        "is_approved": bool(proba_prediction[0, 1] >= 0.5),
        "loan_approval_probability": round(proba_prediction[0, 1], 2),
    }
    logger.info(f"Prediction result: {result}")
    return result
