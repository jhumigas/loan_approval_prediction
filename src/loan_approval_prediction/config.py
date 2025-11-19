from pydantic import Field
from pydantic_settings import BaseSettings


class ProjectConfig(BaseSettings):
    """
    Configuration class for the Loan Approval Prediction project.
    This class uses Pydantic's BaseSettings to manage configuration settings.
    """

    # Project name
    project_name: str = Field(
        default="Loan Approval Prediction", description="The name of the project."
    )

    # Environment
    environment: str = Field(
        default="development",
        description="The environment in which the application is running.",
    )

    # Artifact directory
    artifact_dir: str = Field(
        default="artifacts", description="Directory where artifacts will be stored."
    )

    # Data directory
    data_dir: str = Field(
        default="data", description="Directory where data files are stored."
    )

    # Model directory
    model_dir: str = Field(
        default="models", description="Directory where trained models are saved."
    )
    # Model path
    model_file: str = Field(
        default="model.bin", description="Path to the trained model file."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
