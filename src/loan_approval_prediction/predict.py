# TODO: loads model, runs inference
import pickle
from loguru import logger
from loan_approval_prediction.config import ProjectConfig
import sklearn
import polars as pl


def load_model(model_path: str):
    """Load the trained model from the specified path."""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        logger.info("Model loaded successfully")
    return model


def predict_proba(model, input_data: dict):
    """Run inference using the loaded model and input data."""
    input_df = pl.DataFrame([input_data])
    prediction = model.predict_proba(input_df)
    return prediction


if __name__ == "__main__":
    config = ProjectConfig()
    model_path = f"{config.model_dir}/model.pkl"

    model = load_model(model_path)

    # Example input data for prediction
    sample_input = {
        "income": 62098,
        "credit_score": 689,
        "loan_amount": 19217,
        "years_employed": 29,
        "points": 65,
    }

    prediction = predict_proba(model, sample_input)
    logger.info(f"Prediction result: {prediction}")
