import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.utils.utils import prepare_data


class Linear:
    """
    Linear Regression model for audio classification.

    This class uses scikit-learn's LinearRegression to perform binary classification.
    """
    def __init__(self):
        self.lin_reg = LinearRegression()

    def train(self, train_dataloader: DataLoader, model_name: str):
        """
        Trains the Linear Regression model.

        Parameters:
        - train_dataloader (DataLoader): DataLoader for training data.
        - model_name (str): Name of the model for saving.

        Returns:
        - None
        """
        X_train, y_train = prepare_data(train_dataloader)
        self.lin_reg.fit(X_train, y_train)
        with open(f'models/{model_name}.pkl','wb+') as f:
            pickle.dump(self.lin_reg, f)

    def eval(self, test_dataloader: DataLoader)-> (str, dict):
        """
        Evaluates the Linear Regression model.

        Parameters:
        - test_dataloader (DataLoader): DataLoader for test data.

        Returns:
        - tuple: A tuple containing the accuracy and classification report.
        """
        X_test, y_test = prepare_data(test_dataloader)
        y_pred_cont = self.lin_reg.predict(X_test)
        y_pred = (y_pred_cont >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(report)
        return accuracy, report
