import sqlite3
import pandas as pd

class DataFetching:
    def __init__(self, database_path_1, query_1, database_path_2, query_2):
        self.database_path_1 = database_path_1
        self.query_1 = query_1
        self.database_path_2 = database_path_2
        self.query_2 = query_2

    def fetch_data(self):
        conn1 = sqlite3.connect(self.database_path_1)
        df1 = pd.read_sql_query(self.query_1, conn1)
        conn1.close()
        
        conn2 = sqlite3.connect(self.database_path_2)
        df2 = pd.read_sql_query(self.query_2, conn2)
        conn2.close()
        
        return df1, df2
