import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.utils.utils import prepare_data


class RandomForest:
    """Random Forest model for audio classification.

    This class uses scikit-learn's RandomForestClassifier."""
    def __init__(self, n_estimators:int =50, random_state:int =42):
        self.rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10,
                                                    max_features="sqrt", min_samples_split=50, min_samples_leaf=20,
                                                    max_samples=0.5)

    def train(self, train_dataloader: DataLoader, model_name: str):
        """Trains the Random Forest model.

        Parameters:
        - train_dataloader (DataLoader): DataLoader for training data.
        - model_name (str): Name of the model for saving.

        Returns:
        - None
        """
        X_train, y_train = prepare_data(train_dataloader)
        self.rf_classifier.fit(X_train, y_train)
        with open(f'models/{model_name}.pkl','wb+') as f:
            pickle.dump(self.rf_classifier, f)

    def eval(self, test_dataloader: DataLoader)-> (str, dict):
        """Evaluates the Random Forest model.

        Parameters:
        - test_dataloader (DataLoader): DataLoader for test data.

        Returns:
        - tuple: A tuple containing the accuracy and classification report.
        """
        X_test, y_test = prepare_data(test_dataloader)
        y_pred = self.rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(report)
        return accuracy, report
