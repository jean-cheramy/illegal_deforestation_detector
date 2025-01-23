import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV


class RandomForest:
    def __init__(self, n_estimators=50, random_state=42):
        self.rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=10,
                                                    max_features="sqrt", min_samples_split=50, min_samples_leaf=20,
                                                    max_samples=0.5)

    def prepare_data(self, dataloader):
        print("preparing the data...")
        all_data = []
        all_labels = []
        for batch_data, batch_labels in dataloader:
            all_data.append(batch_data)
            all_labels.append(batch_labels)

        # Concatenate all batches into single tensors
        concat_data = torch.cat(all_data, dim=0)
        concat_labels = torch.cat(all_labels, dim=0)
        concat_data = concat_data.view(concat_data.size(0), -1)
        print(f"{len(concat_data)} records")
        return concat_data.numpy(), concat_labels.numpy()

    def train(self, train_dataloader):
        X_train, y_train = self.prepare_data(train_dataloader)
        print("training...")

        # rf_param_dist = {
        #     'n_estimators': [10, 50, 100],
        #     'max_depth': [5, 10, 15],
        #     'max_features': ['sqrt', 'log2'],
        #     'min_samples_split': [5, 10, 20],
        #     'min_samples_leaf': [2, 4, 8]
        # }
        #
        # random_search = RandomizedSearchCV(self.rf_classifier, param_distributions=rf_param_dist, n_iter=20, cv=3,
        #                                    scoring='accuracy', random_state=42)
        # random_search.fit(X_train, y_train)

        # print("Best parameters:", random_search.best_params_)
        # exit()

        self.rf_classifier.fit(X_train, y_train)

    def eval_model(self, test_dataloader):
        X_test, y_test = self.prepare_data(test_dataloader)
        print("testing...")
        y_pred = self.rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))