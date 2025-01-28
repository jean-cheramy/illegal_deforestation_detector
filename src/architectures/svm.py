import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from src.data import prepare_data


class SVM:
    def __init__(self):
        self.svm_model = SVC(kernel="rbf")

    def train(self, train_dataloader):
        X_train, y_train = prepare_data(train_dataloader)
        self.svm_model.fit(X_train, y_train)

    def eval(self, test_dataloader):
        X_test, y_test = prepare_data(test_dataloader)
        y_pred = self.svm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
