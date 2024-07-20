import argparse
import yaml
from .data_fetching import DataFetching
""" from .data_cleaning import DataCleaning
from .feature_engineering import FeatureEngineering
from .model_training import ModelTraining
from .model_initial_eval import ModelEvaluation
from .model_fine_tuning import ModelFineTuning
from .model_final_eval import ModelEvaluation
import joblib """

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Fetch data
    data_fetcher = DataFetching(
        config['database_path_1'], config['query_1'],
        config['database_path_2'], config['query_2']
    )
    df1, df2 = data_fetcher.fetch_data()

    """ # Clean and merge data
    processor = DataCleaning(config)
    df1 = processor.clean_data(df1)
    df2 = processor.clean_data(df2)
    df = processor.merge_data(df1, df2)

    # Preprocess data
    df = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.split_data(df, 'Daily Solar Panel Efficiency')

    # Feature engineering
    engineer = FeatureEngineering(config)
    X_train = engineer.engineer_features(X_train)
    X_test = engineer.engineer_features(X_test)

    # Train initial model
    trainer = ModelTraining(config)
    model = trainer.train_model(X_train, y_train)
    joblib.dump(model, 'models/trained_model.joblib')

    # Evaluate initial model
    evaluator = ModelEvaluation(config)
    initial_accuracy, initial_report = evaluator.evaluate_model(model, X_test, y_test)
    print(f"Initial Model Accuracy: {initial_accuracy}")
    print(f"Initial Model Classification Report:\n{initial_report}")

    # Fine-tune model
    fine_tuner = ModelFineTuning(config)
    best_model = fine_tuner.fine_tune_model(X_train, y_train)
    joblib.dump(best_model, 'models/best_model.joblib')

    # Evaluate fine-tuned model
    final_accuracy, final_report = evaluator.evaluate_model(best_model, X_test, y_test)
    print(f"Fine-Tuned Model Accuracy: {final_accuracy}")
    print(f"Fine-Tuned Model Classification Report:\n{final_report}") """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the end-to-end machine learning pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    main(args.config)