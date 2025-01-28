import torch
import torch.nn as nn

from src.eval import model_eval


class TinyDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TinyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_tiny_dnn(train_dataloader, test_dataloader, feature_length, num_epochs):
    model = TinyDNN(input_size=(13 + 128) * feature_length, hidden_size=6, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.view(batch_features.size(0), -1)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("trained!")
    model_eval(model, test_dataloader)
