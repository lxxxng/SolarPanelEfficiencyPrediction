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
    # some numeric features are of object dtype, convert to numeric dtype
    
    # 2.1 Define columns to convert to numeric in weather data
    weather_numeric_columns = [
        'Daily Rainfall Total (mm)',
        'Highest 30 Min Rainfall (mm)',
        'Highest 60 Min Rainfall (mm)',
        'Highest 120 Min Rainfall (mm)',
        'Min Temperature (deg C)',
        'Maximum Temperature (deg C)',
        'Min Wind Speed (km/h)',
        'Max Wind Speed (km/h)'
    ]
    
    # 2.2 Define columns to convert to numeric in air data
    air_numeric_columns = [
        'pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central',
        'psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central'
    ]
    
    processor = DataPreprocessing(config)
    weather_df = processor.convert_datatype_numeric(weather_numeric_columns, weather_df) 
    air_df = processor.convert_datatype_numeric(air_numeric_columns, air_df) 
    
    # 2.3 convert fahrenheit to celcius
    weather_df = processor.fahrenheit_to_celcius('Wet Bulb Temperature (deg F)', weather_df)
    
    # 2.4 remove duplicates
    weather_df = weather_df.drop_duplicates()
    air_df = air_df.drop_duplicates()
    
    # 2.5 group by and fill based on date
    air_df = processor.group_by_and_fill('date', air_df)
    
    # 2.6. handle missing values
    weather_df = processor.handle_missing_values(weather_df)
    air_df = processor.handle_missing_values(air_df)

    # 2.8 merge datasets based on date
    merged_df = processor.merge_data(weather_df, air_df, 'date')
    
    # feature engineering
    engineer = FeatureEngineering(config)
    merged_df = engineer.engineer_features(merged_df)
    
    # feature selection
    to_drop = ['data_ref_x', 'data_ref_y', 'Highest 30 Min Rainfall (mm)', \
            'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)', 'date']
    merged_df.drop(columns=to_drop, inplace=True)
    
    #2.9 scaling numerical features
    merged_df = processor.scale_data(merged_df)
    
    # 2.10 Mapping categorical features 
    merged_df = processor.map_categorical(merged_df)
    
    # encoding categorical features
    merged_df = processor.label_encode_data(merged_df)
    merged_df = processor.one_hot_encode_data(merged_df)

    # Split into train 80% and test set 20% with stratified sampling
    X_train, X_test, y_train, y_test = processor.split_data(merged_df, 'Daily Solar Panel Efficiency')

    # Train initial models
    trainer = ModelTraining(config)
    models = trainer.train_models(X_train, y_train)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    for model_name, model in models.items():
        joblib.dump(model, f'models/{model_name}_trained_model.joblib')

    # Evaluate initial models
    evaluator = ModelEvaluation(config)
    initial_results = evaluator.evaluate_models(models, X_test, y_test)
    print("Before fine tune \n")
    evaluator.print_results(initial_results)

    # Fine-tune models
    print("--------------------------- Fine tuning -----------------------------------")
    fine_tuner = ModelFineTuning(config)
    best_models, best_params = fine_tuner.fine_tune_models(models, X_train, y_train)
    
    for model_name, model in best_models.items():
        joblib.dump(model, f'models/{model_name}_best_model.joblib')
        
    print("--------------------------- Fine tuning completed ! -----------------------------------")
        
    print("Best Parameters for Fine-Tuned Models:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")

    # Evaluate fine-tuned models
    final_results = evaluator.evaluate_models(best_models, X_test, y_test)
    
    print("\n Results after fine tuning \n")
    evaluator.print_results(final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the end-to-end machine learning pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    main(args.config)