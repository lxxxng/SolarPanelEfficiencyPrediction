from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class ModelFineTuning:
    def __init__(self, config):
        self.config = config

    def fine_tune_model(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        return best_estimator, best_params


    def fine_tune_models(self, models, X_train, y_train):
        best_models = {}
        best_params = {}

        for model_name, model in models.items():
            if model_name == 'RandomForestClassifier':
                param_grid = self.config['model_fine_tuning']['RandomForestClassifier']['param_grid']
            elif model_name == 'SVC':
                param_grid = self.config['model_fine_tuning']['SVC']['param_grid']
            elif model_name == 'LogisticRegression':
                param_grid = self.config['model_fine_tuning']['LogisticRegression']['param_grid']
            else:
                raise ValueError("Unsupported algorithm specified in config.")
            
            best_model, best_param = self.fine_tune_model(model, param_grid, X_train, y_train)
            best_models[model_name] = best_model
            best_params[model_name] = best_param

        return best_models, best_params

