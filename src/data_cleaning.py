import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self, config):
        self.config = config

    def clean_data(self, df):
        df['date'] = pd.to_datetime(df['date'])
        if self.config['preprocessing']['dropna']:
            df = df.dropna()

        if self.config['preprocessing']['imputation_strategy']:
            imputer = SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy'])
            df[df.columns] = imputer.fit_transform(df)

        return df

    def merge_data(self, df1, df2):
        df = pd.merge(df1, df2, on='date', how='inner')
        return df

    def preprocess_data(self, df):
        if self.config['preprocessing']['scaling'] == 'standard':
            scaler = StandardScaler()
        elif self.config['preprocessing']['scaling'] == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None

        if scaler:
            df[df.columns] = scaler.fit_transform(df)

        if self.config['preprocessing']['encoding'] == 'onehot':
            df = pd.get_dummies(df)
        elif self.config['preprocessing']['encoding'] == 'label':
            encoder = OneHotEncoder()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = encoder.fit_transform(df[col])

        return df

    def split_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test