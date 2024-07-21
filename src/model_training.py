from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ModelTraining:
    def __init__(self, config):
        self.config = config

    def train_models(self, X_train, y_train):
        algorithms = self.config['model_training']['algorithms']
        models = {}

        for algorithm in algorithms:
            if algorithm == 'RandomForestClassifier':
                model = RandomForestClassifier(random_state=42)
            elif algorithm == 'SVC':
                model = SVC(random_state=42)
            elif algorithm == 'LogisticRegression':
                model = LogisticRegression(random_state=42)
            else:
                raise ValueError("Unsupported algorithm specified in config.")
            
            model.fit(X_train, y_train)
            models[algorithm] = model
            
        return models
