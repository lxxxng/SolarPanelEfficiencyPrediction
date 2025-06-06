﻿# EDA & end to end machine pipeline
 
## Overview of the Submitted Folder and Folder Structure
The submitted folder contains the following structure:

```python
.
├── .github
│ └── workflows
│   └── github-actions.yml
├── src
│ ├── config.yaml
│ ├── data_fetching.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── init.py
│ ├── main.py
│ ├── model_evaluation.py
│ ├── model_fine_tuning.py
│ ├── model_training.py
├── eda.ipynb
├── README.md
├── requirements.txt
└── run.sh
```

## Instruction to run the pipeline
- github-actions.yml file will automate the installation of libraries in 'requirements.txt'
followed by running the bash script 'run.sh', which will run the files in 'src' folder and start running the pipeline

OR 

-  manually enter into your terminal 
    - python -m pip install --upgrade pip
    - pip install -r requirements.txt
    - python -m src.main --config src/config.yaml', (this is in the bash script) 
The main.py file will start running the pipeline and call other classes and functions. 

- Note: Database is not uploaded to github based on the requirements, so you need to create a folder name 'date' at root
and add in the 2 datasets, name them weather.db and air_quality.db

## To modify parameters
- In the 'src' folder, there is a file named config.yaml, changes can be performed to database path, preprocessing methods,
choice of algorithms and hyper parameters to fine tune the models based on your needs. 


## Pipeline flow
1. Fetch data based on the paths described in config.yaml
2. Preprocessing
    - cleaning of data (missing values, converting dtypes, remove duplicates, merge datasets etc.)
3. Feature engineering
4. Feature selection (removing unwanted columns based on eda)
5. Scaling (standard scaling) & Encoding (one-hot and label)
6. Split into train (80) and test set (20) (stratified)
7. Train models
8. Inital evaluation
9. Fine-tune models
10. Evaluate fine-tuned models

## EDA overview & feature engineering
- I performed some data cleaning and merged the two datasets together to have a comprehensive set of data.
- Correlation analysis were used to identify numerical features that are correlated to each other, which could be redundant.
- Then I carried out visualization for each feature, to see how they impact solar panel efficiency and interact with different features. 
- The more siginificant features are: 
1. Daily Rainfall Total
2. Min temperature
3. Max temperature
4. Min Wind Speed
5. Max Wind Speed
6. Sunshine Duration
7. Cloud Cover
8. Relative humidity 

- Feature engineering --> Based on eda, I created new features to visualize its impact on efficiency. 
1. Temperature range
    - Created by calculating the difference between the maximum and minimum temperatures. This feature helps in understanding the temperature stability which might impact solar panel efficiency.
2. Wind speed range
    - Created by calculating the difference between the maximum and minimum wind speeds. This feature can provide insights into wind variability and its impact on solar panel efficiency.
3. Month 
    - The month of the year was included as a feature to capture seasonal variations that might affect solar panel performance.
4. psi_mean
    - psi of all regions are highly correlated and show similar patterns, so I aggreagted them together. 
5. pm25_mean
    - pm25 of all regions are highly correlated and show similar patterns, so I aggreagted them together.


## Feature Processing Summary

### Weather Dataset

| Feature                        | Processing Steps                                         |
|--------------------------------|----------------------------------------------------------|
| `data_ref`                     | Dropped as it is not relevant for modeling               |
| `date`                         | Converted to datetime, extracted month, outerjoin, dropped |
| `Daily Rainfall Total (mm)`    | Converted to float64, imputed missing values, normalized |
| `Highest 30 Min Rainfall (mm)` | dropped, highly correlated                               |
| `Highest 60 Min Rainfall (mm)` | dropped, highly correlated                               |
| `Highest 120 Min Rainfall (mm)`| dropped, highly correlated                               |
| `Min Temperature (deg C)`      | Converted to float64, imputed missing values, feature engineering, normalized |
| `Maximum Temperature (deg C)`  | Converted to float64, imputed missing values, feature engineering, normalized |
| `Min Wind Speed (km/h)`        | Converted to float64, imputed missing values, feature engineering, normalized |
| `Max Wind Speed (km/h)`        | Converted to float64, imputed missing values, feature engineering, normalized |
| `Sunshine Duration (hrs)`      | Converted to float64, imputed missing values, normalized |
| `Cloud Cover (%)`              | Converted to float64, imputed missing values, normalized |
| `Relative Humidity (%)`        | Converted to float64, imputed missing values, normalized |
| `Air Pressure (hPa)`           | Converted to float64, normalized                         |
| `Dew Point Category`           | One-hot encoded                                          |
| `Wind Direction`               | One-hot encoded                                          |
| `Daily Solar Panel Efficiency` | Target variable, label encoded                           |
| `Wet Bulb Temperature (deg C)` | Converted to float64, imputed missing values, normalized |

### Air Quality Dataset

| Feature        | Processing Steps                                         |
|----------------|----------------------------------------------------------|
| `data_ref`     | Dropped as it is not relevant for modeling               |
| `date`         | Converted to datetime, outerjoin, extracted month, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |
| `pm25_north`   | Converted to float64, imputed missing values, aggregation, dropped |


## Explanation of Choice of Models

### RandomForestClassifier
The RandomForestClassifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees. It helps with multiclass classification by:
- **Robustness**: Combines the predictions of multiple decision trees to reduce overfitting.
- **Feature Importance**: Provides insights into feature importance, which can be useful for understanding the data.
- **Reduction of Overfitting**: Less prone to overfitting compared to individual decision trees.

### DecisionTreeClassifier
The DecisionTreeClassifier is a simple and intuitive model that splits the data based on feature values to make predictions. It helps with multiclass classification by:
- **Simplicity**: Easy to interpret and visualize.
- **Non-Linear Relationships**: Captures non-linear relationships between features.
- **Multiclass Capability**: Naturally extends to multiclass classification problems.

### SVC (Support Vector Classifier)
The SVC is a powerful classifier that finds the hyperplane that best separates the classes in the feature space. It helps with multiclass classification by:
- **Margin Maximization**: Finds the hyperplane that maximizes the margin between classes.
- **Kernel Trick**: Uses kernel functions to handle non-linear classification.
- **One-vs-One Strategy**: Implements a one-vs-one strategy for multiclass classification, creating multiple binary classifiers.

### LogisticRegression
The LogisticRegression model predicts the probability of a categorical dependent variable. It helps with multiclass classification by:
- **Baseline Performance**: Often serves as a good baseline model for comparison.
- **Simplicity**: Simple to implement and interpret.
- **One-vs-Rest Strategy**: Implements a one-vs-rest strategy for multiclass classification.

### MLP (Multi-Layer Perceptron)
The MLP is a type of neural network that consists of multiple layers of nodes. It helps with multiclass classification by:
- **Complex Patterns**: Capable of capturing complex patterns and interactions in the data.
- **Non-Linear Relationships**: Handles non-linear relationships through activation functions.
- **Softmax Activation**: Uses softmax activation in the output layer for multiclass classification.

## Evaluation of Models

### Evaluation Metrics
To evaluate the models, the following metrics are used:

- **Accuracy**: The proportion of correct predictions out of the total predictions. It provides a general sense of how well the model performs.
- **Classification Report**: Includes precision, recall, and F1-score for each class:
  - **Precision**: The number of true positive results divided by the number of positive results predicted by the classifier.
  - **Recall**: The number of true positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
  - **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **Confusion Matrix**: A matrix that shows the number of correct and incorrect predictions made by the model (TP, TN, FP, FN), broken down by each class.

### Results Before Fine-Tuning

| Model                    | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|--------------------------|----------|-----------------------|--------------------|----------------------|
| RandomForestClassifier   | 85.63%   | 84.76%                | 81.89%             | 83.15%               |
| DecisionTreeClassifier   | 72.81%   | 68.46%                | 68.86%             | 68.65%               |
| SVC                      | 77.34%   | 76.90%                | 70.19%             | 72.53%               |
| LogisticRegression       | 75.00%   | 73.74%                | 66.96%             | 69.21%               |
| MLP                      | 75.78%   | 73.76%                | 69.81%             | 71.23%               |

### Results After Fine-Tuning

| Model                    | Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|--------------------------|----------|-----------------------|--------------------|----------------------|
| RandomForestClassifier   | 85.94%   | 85.14%                | 82.24%             | 83.51%               |
| DecisionTreeClassifier   | 84.69%   | 84.04%                | 80.98%             | 82.32%               |
| SVC                      | 77.19%   | 76.43%                | 70.41%             | 72.60%               |
| LogisticRegression       | 75.16%   | 73.97%                | 67.06%             | 69.34%               |
| MLP                      | 80.31%   | 79.16%                | 75.17%             | 76.75%               |

### Best Parameters for Fine-Tuned Models

| Model                    | Best Parameters                                                                                                                                             |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RandomForestClassifier   | {'bootstrap': False, 'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}                                                   |
| DecisionTreeClassifier   | {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 30, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 20, 'splitter': 'best'} |
| SVC                      | {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}                                                                                                  |
| LogisticRegression       | {'C': 1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'sag'}                                                                                                  |
| MLP                      | {'activation': 'tanh', 'hidden_layer_sizes': [150], 'learning_rate_init': 0.01, 'max_iter': 200, 'solver': 'sgd'}                                             |

