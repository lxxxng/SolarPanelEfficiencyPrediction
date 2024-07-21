import argparse
import yaml
import os
import joblib
import pandas as pd
import numpy as np
from .data_fetching import DataFetching
from .data_preprocessing import DataPreprocessing
from .feature_engineering import FeatureEngineering
from .model_training import ModelTraining
from .model_fine_tuning import ModelFineTuning
from .model_evaluation import ModelEvaluation

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 1. Fetch data
    data_fetcher = DataFetching(
        config['weather_path'], config['query_1'],
        config['air_path'], config['query_2']
    )
    
    weather_df, air_df = data_fetcher.fetch_data()
    
    # 2. Preprocess data
    processor = DataPreprocessing(config)
    merged_df = processor.preprocess_data(weather_df, air_df)
    
    # 3. feature engineering
    engineer = FeatureEngineering(config)
    merged_df = engineer.engineer_features(merged_df)
    
    # 4. feature selection
    # These features are removed based on analysis in eda
    to_drop = ['data_ref_x', 'data_ref_y', 'Highest 30 Min Rainfall (mm)', \
            'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)', 'date']
    merged_df.drop(columns=to_drop, inplace=True)
    
    # 5 scaling numerical features & encoding categorical features 
    merged_df = processor.scale_data(merged_df)
    merged_df = processor.label_encode_data(merged_df)
    merged_df = processor.one_hot_encode_data(merged_df)

    # 6 Split into train set 80% and test set 20% with stratified sampling
    X_train, X_test, y_train, y_test = processor.split_data(merged_df, 'Daily Solar Panel Efficiency')

    # 7 Train initial models
    trainer = ModelTraining(config)
    models = trainer.train_models(X_train, y_train)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    for model_name, model in models.items():
        joblib.dump(model, f'models/{model_name}_trained_model.joblib')     # dumping trained models into a folder

    # 8 Evaluate initial models
    evaluator = ModelEvaluation(config)
    initial_results = evaluator.evaluate_models(models, X_test, y_test)
    print("\n Before fine tune \n")
    evaluator.print_results(initial_results)

    # 9 Fine-tune models
    print("\n--------------------------------- Fine tuning ------------------------------------")
    fine_tuner = ModelFineTuning(config)
    best_models, best_params = fine_tuner.fine_tune_models(models, X_train, y_train)
    
    for model_name, model in best_models.items():
        joblib.dump(model, f'models/{model_name}_best_model.joblib')    # dumping fine tuned models into a folder
        
    print("\n--------------------------- Fine tuning completed ! -----------------------------------")
        
    print("\n Best Parameters for Fine-Tuned Models:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")

    # 10 Evaluate fine-tuned models
    final_results = evaluator.evaluate_models(best_models, X_test, y_test)
    
    print("\n Results after fine tuning \n")
    evaluator.print_results(final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the end-to-end machine learning pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    main(args.config)