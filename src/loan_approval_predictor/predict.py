import argparse
import pickle
from loguru import logger
from loan_approval_predictor.config import ProjectConfig
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
    config = ProjectConfig.from_yaml("config.yml")

    model = load_model(config.model_path)

    argparser = argparse.ArgumentParser(
        description="Run inference on sample input data."
    )
    argparser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file."
    )
    args = argparser.parse_args()
    sample_input = pl.read_json(args.input).to_dicts()[0]

    prediction = predict_proba(model, sample_input)
    logger.info(f"Prediction result: {prediction}")
