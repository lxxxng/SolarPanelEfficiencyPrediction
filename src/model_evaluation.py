from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, matrix

    def evaluate_models(self, models, X_test, y_test):
        results = {}
        for model_name, model in models.items():
            accuracy, report, matrix = self.evaluate_model(model, X_test, y_test)
            results[model_name] = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': matrix
            }
        return results
    
    def print_results(self, results):
        for model_name, result in results.items():
            print(f"{model_name} Accuracy: {result['accuracy']}")
            print(f"{model_name} Classification Report:\n")
            for label, metrics in result['report'].items():
                if isinstance(metrics, dict):
                    print(f"  Label {label}:")
                    for metric_name, value in metrics.items():
                        print(f"    {metric_name}: {value}")
                    print("\n")
                else:
                    print(f"  {label}: {metrics}")
            print(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}")
            print()
            print("----------------------------------------------------------------------------------")
