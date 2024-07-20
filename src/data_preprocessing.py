import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, config):
        self.config = config

    # Convert columns to numeric
    def convert_datatype_numeric(self, columns, df):    
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df 
    
    def convert_datatype_datetime(self, column, df):    
        df[column] = pd.to_datetime(df[column], format='%d/%m/%Y')
        return df 
    
    # convert deg F to deg C
    def fahrenheit_to_celcius(self, column, df):
        df[column] = (df[column] - 32) * 5.0/9.0
        return df 
    
    def group_by_and_fill(self, column, df):
        # Group by the specified day and fill missing values using forward and backward fill
        df = df.groupby(column).apply(lambda x: x.ffill().bfill()).drop_duplicates(column).reset_index(drop=True)
        return df
    
    # hadnle missing values
    def handle_missing_values(self, df):
        num_imputation_strategy = self.config['preprocessing']['num_imputation_strategy']
        cat_imputation_strategy = self.config['preprocessing']['cat_imputation_strategy']

        # Separate numerical and categorical columns
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        # Impute numerical columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy=num_imputation_strategy)
            df[num_cols] = num_imputer.fit_transform(df[num_cols])

        # Impute categorical columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy=cat_imputation_strategy)
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        return df

    # merge the 2 datasets using outer join
    def merge_data(self, df1, df2, column):
        self.convert_datatype_datetime(column, df1)
        self.convert_datatype_datetime(column, df2)
        merged_df = pd.merge(df1, df2, on=column, how='outer')
        return merged_df

    # mapping for  categorical feaetures
    def map_categorical(self, df):
        wind_direction_map = {
            'N.': 'N', 
            'W': 'W',
            'S': 'S',
            'E': 'E',
            'east': 'E',
            'NORTHEAST': 'NE',
            'NW': 'NW',
            'NE': 'NE',
            'SE': 'SE',
            'Southward': 'S',
            'W.': 'W',
            'southeast': 'SE',
            'SW': 'SW',
            'N': 'N',
            'Northward': 'N',
            'SOUTHEAST': 'SE',
            'northwest': 'NW',
            'west': 'W',
            'NORTH': 'N',
            'south': 'S',
            'NE.': 'NE',
            'SE.': 'SE',
            'NORTHWEST': 'NW',
            'northeast': 'NE',
            'SW.': 'SW',
            'north': 'N',
            'SOUTH': 'S',
            'E.': 'E',
            'S.': 'S',
            'NW.': 'NW',
            'WEST': 'W',
            'EAST': 'E'
        }
        

        # Mapping dictionary for 'dew point category'
        label_map = {
            'vh': 'High',
            'very high': 'High',
            'low': 'Low',
            'vl': 'Low',
            'very low': 'Low',
            'high': 'High',
            'moderate': 'Moderate',
            'm': 'Moderate',
            'h': 'High',
            'Extreme': 'Extreme',
            'minimal': 'Low',
            'normal': 'Moderate',
            'high level': 'High',
            'below average': 'Low',
            'l': 'Low'  
        }
        
        # apply the mapping
        df['Dew Point Category'] = df['Dew Point Category'].map(label_map)
        df['Wind Direction'] = df['Wind Direction'].map(wind_direction_map)
        
        return df
        
    # scale numeric data
    def scale_data(self, df):
        if self.config['preprocessing']['scaling'] == 'standard':
            scaler = StandardScaler()
        elif self.config['preprocessing']['scaling'] == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None

        if scaler:
            num_cols = df.select_dtypes(include=['number']).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

    # label encode ordinal data
    def label_encode_data(self, df):
        label_cols = self.config['preprocessing']['label_encode_columns']
        for col in label_cols:
            if col == 'Daily Solar Panel Efficiency':
                df[col] = df[col].map({'Low': 0, 'Medium': 1, 'High': 2})
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        return df

    # label encode nominal data
    def one_hot_encode_data(self, df):
        one_hot_cols = self.config['preprocessing']['one_hot_encode_columns']
        df = pd.get_dummies(df, columns=one_hot_cols)
        return df

    def split_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(\
            X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test