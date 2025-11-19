from loan_approval_prediction import train
from loguru import logger


if __name__ == "__main__":
    from loan_approval_prediction.config import ProjectConfig

    config = ProjectConfig.from_yaml("config.yml")

    # Load data
    raw_data = train.load_data(config.data_path)

    # Data preparation
    full_x, full_y = train.split_label(
        raw_data, target_col=config.data_config.target_column
    )

    # Split dataset
    x_raw_train, x_raw_val, x_raw_test = train.split_dataset(
        full_x,
        val_size=config.evaluation_config.val_ratio,
        test_size=config.evaluation_config.test_ratio,
        seed=config.random_seed,
    )
    y_raw_train, y_raw_val, y_raw_test = train.split_dataset(
        full_y,
        val_size=config.evaluation_config.val_ratio,
        test_size=config.evaluation_config.test_ratio,
        seed=config.random_seed,
    )

    # Feature extraction and cleaning
    x_train, y_train = train.clean_dataset(
        x_raw_train, y_raw_train, config.data_config.numerical_columns
    )
    x_val, y_val = train.clean_dataset(
        x_raw_val, y_raw_val, config.data_config.numerical_columns
    )
    x_test, y_test = train.clean_dataset(
        x_raw_test, y_raw_test, config.data_config.numerical_columns
    )

    # Train model
    model = train.create_model_instance(
        C=config.model_parameters.C,
        max_iter=config.model_parameters.max_iter,
        solver=config.model_parameters.solver,
        random_state=config.model_parameters.random_seed,
    )
    model = train.train_model(model, x_train, y_train)

    # Evaluate model on validation and training datasets
    logger.info("Evaluating model on training and validation datasets.")
    logger.info("Training dataset evaluation:")
    roc_auc_train = train.evaluate_model(model, x_train, y_train)
    logger.info(f"Train ROC AUC: {roc_auc_train}")

    logger.info("Validation dataset evaluation:")
    roc_auc_val = train.evaluate_model(model, x_val, y_val)
    logger.info(f"Validation ROC AUC: {roc_auc_val}")

    # Train on full train dataset
    logger.info("Retraining model on the full training dataset.")
    x_full_train, y_full_train = train.concatenate_datasets(
        x_train, y_train, x_val, y_val
    )
    model_final = train.train_model(model, x_full_train, y_full_train)
    logger.info("Evaluating model on full training and test datasets.")
    logger.info("Full training dataset evaluation:")
    roc_auc_full_train = train.evaluate_model(model_final, x_full_train, y_full_train)
    logger.info(f"Train ROC AUC: {roc_auc_full_train}")

    logger.info("Test dataset evaluation:")
    roc_auc_test = train.evaluate_model(model_final, x_test, y_test)
    logger.info(f"Validation ROC AUC: {roc_auc_test}")

    # Save model
    train.save_model(model_final, config.model_path)
