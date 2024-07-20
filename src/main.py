import argparse
import yaml
import pandas as pd
from .data_fetching import DataFetching
from .data_preprocessing import DataPreprocessing
from .feature_engineering import FeatureEngineering
""" 
from .model_training import ModelTraining
from .model_initial_eval import ModelEvaluation
from .model_fine_tuning import ModelFineTuning
from .model_final_eval import ModelEvaluation
import joblib """

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
    
    #2.9 scaling numerical features
    merged_df = processor.scale_data(merged_df)
    
    # 2.10 Mapping categorical features 
    merged_df = processor.map_categorical(merged_df)
    
    # 2.11 encoding categorical features
    merged_df = processor.label_encode_data(merged_df)
    merged_df = processor.one_hot_encode_data(merged_df)
    
    print(merged_df.info())

    """ 
    X_train, X_test, y_train, y_test = processor.split_data(df, 'Daily Solar Panel Efficiency')

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