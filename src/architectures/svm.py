import pickle

from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.utils.utils import prepare_data


class SVM:
    """Support Vector Machine model for audio classification.

    This class uses scikit-learn's SVC with an RBF kernel."""
    def __init__(self):
        self.svm_model = SVC(kernel="rbf")

    def train(self, train_dataloader: DataLoader, model_name: str):
        """Trains the SVM model.

        Parameters:
        - train_dataloader (DataLoader): DataLoader for training data.
        - model_name (str): Name of the model for saving.

        Returns:
        - None
        """
        X_train, y_train = prepare_data(train_dataloader)
        self.svm_model.fit(X_train, y_train)
        with open(f'models/{model_name}.pkl','wb+') as f:
            pickle.dump(self.svm_model, f)


    def eval(self, test_dataloader: DataLoader)-> (str, dict):
        """Evaluates the SVM model.

        Parameters:
        - test_dataloader: DataLoader for test data.

        Returns:
        - tuple: A tuple containing the accuracy and classification report.
        """
        X_test, y_test = prepare_data(test_dataloader)
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(report)
        return accuracy, report

