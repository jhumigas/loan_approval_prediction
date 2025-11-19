from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class DataConfig(BaseSettings):
    """
    Data configuration for the Loan Approval Prediction project.
    """

    numerical_columns: list[str] = Field(
        ..., description="List of numerical feature column names."
    )
    categorical_columns: list[str] = Field(
        ..., description="List of categorical feature column names."
    )
    target_column: str = Field(..., description="Name of the target variable column.")


class ModelParameters(BaseSettings):
    """
    Model parameters for the Loan Approval Prediction model.
    """

    random_seed: int = Field(..., description="Random seed for model reproducibility.")
    max_iter: int = Field(
        ..., description="Maximum number of iterations for the model training."
    )
    C: float = Field(
        ..., description="Inverse of regularization strength for the model."
    )
    solver: str = Field(
        ..., description="Algorithm to use in the optimization problem."
    )


class EvaluationConfig(BaseSettings):
    """
    Evaluation configuration for the Loan Approval Prediction model.
    """

    random_seed: int = Field(
        ..., description="Random seed for evaluation reproducibility."
    )
    val_ratio: float = Field(
        ..., description="Proportion of the dataset to include in the validation split."
    )
    test_ratio: float = Field(
        ..., description="Proportion of the dataset to include in the test split."
    )
    cv_n_splits: int = Field(..., description="Number of folds for cross-validation.")
    evaluation_metric: str = Field(..., description="Metric used for model evaluation.")


class ProjectConfig(BaseSettings):
    """
    Configuration class for the Loan Approval Prediction project.
    This class uses Pydantic's BaseSettings to manage configuration settings.
    """

    project_name: str = Field(..., description="Name of the project.")
    data_path: str = Field(..., description="Paths to the datasets.")
    data_config: DataConfig = Field(
        ..., description="Data configuration such as numerical and categorical columns."
    )
    model_path: str = Field(..., description="Path to save or load the model.")
    model_parameters: ModelParameters = Field(
        ..., description="Model configuration parameters."
    )
    evaluation_config: EvaluationConfig = Field(
        ..., description="Evaluation configuration such as validation and test sizes."
    )
    random_seed: int = Field(..., description="Random seed for reproducibility.")

    @classmethod
    def from_yaml(cls, config_path: str) -> "ProjectConfig":
        with open(config_path, mode="r") as f:
            config_dict = yaml.safe_load(f)
            return cls(**config_dict)
