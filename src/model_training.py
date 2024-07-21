from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class ModelTraining:
    def __init__(self, config):
        self.config = config

    def train_models(self, X_train, y_train):
        algorithms = self.config['model_training']['algorithms']
        models = {}

        for algorithm in algorithms:
            if algorithm == 'RandomForestClassifier':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier(random_state=42)
            elif algorithm == 'SVC':
                model = SVC(kernel='linear', C=1.0, random_state=42)
            elif algorithm == 'LogisticRegression':
                model = LogisticRegression(max_iter=200, random_state=42)
            elif algorithm == 'MLP':
                model = MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
            else:
                raise ValueError("Unsupported algorithm specified in config.")
            
            model.fit(X_train, y_train)
            models[algorithm] = model
            
        return models
