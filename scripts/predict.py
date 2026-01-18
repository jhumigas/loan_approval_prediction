import argparse
from loan_approval_predictor.config import ProjectConfig
from loan_approval_predictor import predict
import polars as pl
from loguru import logger


if __name__ == "__main__":
    config = ProjectConfig.from_yaml("config.yml")

    model = predict.load_model(config.model_path)

    argparser = argparse.ArgumentParser(
        description="Run inference on sample input data."
    )
    argparser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file."
    )
    args = argparser.parse_args()
    sample_input = pl.read_json(args.input).to_dicts()[0]

    prediction = predict.predict_proba(model, sample_input)
    logger.info(f"Prediction result: {prediction}")
