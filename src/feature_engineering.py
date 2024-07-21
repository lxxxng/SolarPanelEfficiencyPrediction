class FeatureEngineering:
    def __init__(self, config):
        self.config = config

    def engineer_features(self, df):
        df['month'] = df['date'].dt.month
    
        psi_columns = ['psi_north', 'psi_south', 'psi_east', 'psi_west', 'psi_central']
        pm25_columns = ['pm25_north', 'pm25_south', 'pm25_east', 'pm25_west', 'pm25_central']
        
        df['psi_mean'] = df[psi_columns].mean(axis=1)
        df['pm25_mean'] = df[pm25_columns].mean(axis=1)
        df.drop(columns=psi_columns + pm25_columns, inplace=True)
        
        df['temp_range'] = df['Maximum Temperature (deg C)'] - df['Min Temperature (deg C)']
        df['wind_speed_range'] = df['Max Wind Speed (km/h)'] - df['Min Wind Speed (km/h)']

        return df