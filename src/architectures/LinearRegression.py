import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from src.data import prepare_data


class Linear:
    def __init__(self):
        self.lin_reg = LinearRegression()

    def train(self, train_dataloader):
        X_train, y_train = prepare_data(train_dataloader)
        self.lin_reg.fit(X_train, y_train)

    def eval(self, test_dataloader):
        X_test, y_test = prepare_data(test_dataloader)
        y_pred_cont = self.lin_reg.predict(X_test)
        y_pred = (y_pred_cont >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
