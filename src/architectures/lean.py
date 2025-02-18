import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """
    Implements early stopping to prevent overfitting during training.

    Parameters:
    - patience (int): Number of epochs to wait for improvement before stopping. Default is 5.
    - min_delta (float): Minimum change in loss to be considered an improvement. Default is 0.001.

    Returns:
    - None
    """
    def __init__(self, patience:int =5, min_delta:int =0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True
        return False


class LightweightAudioNet(nn.Module):
    """
      Lightweight audio classification model using depthwise separable convolutions.

      Parameters:
      - channels (int): Number of input channels.

      Returns:
      - None
      """
    def __init__(self, channels: int):
        """
        Initializes the LightweightAudioNet model.

        Parameters:
        - channels (int): Number of input channels.

        Returns:
        - None
        """
        super(LightweightAudioNet, self).__init__()

        # Using Depthwise Separable Convolutions for efficiency
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, stride=1, padding=1,
                               groups=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1,
                               groups=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv6 = nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1, groups=1)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer to output predictions
        self.fc = nn.Linear(8, 2)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

        # Batch Normalization layers after each convolution for stability
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(32)
        self.bn6 = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Forward pass of the LightweightAudioNet model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # First convolution layer followed by BatchNorm, ReLU, and Dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.conv6(x)))

        # Global Average Pooling
        x = self.pool(x).squeeze(-1)  # Remove the extra dimension after pooling

        # Fully connected layer to output final result
        x = self.fc(x)
        return x


class LEAN:
    """
    Lightweight Audio Neural Network (LEAN) for audio classification.

    Parameters:
    - channels (int): Number of input channels.

    Returns:
    - None
    """
    def __init__(self, channels: int):
        self.model = LightweightAudioNet(channels).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader: DataLoader, model_name: str, epochs:int =10, patience:int =5):
        """
        Trains the LEAN model.

        Parameters:
        - train_loader (DataLoader): DataLoader for training data.
        - model_name (str): Name of the model for saving.
        - epochs (int): Number of training epochs. Default is 10.
        - patience (int): Patience for early stopping. Default is 5.

        Returns:
        - None
        """
        # Convert IterableDataset to list
        dataset_list = list(train_loader.dataset)

        # Manually split dataset into 80% train, 20% validation
        train_size = int(0.8 * len(dataset_list))
        train_data, val_data = dataset_list[:train_size], dataset_list[train_size:]

        # Create new DataLoaders
        train_loader = DataLoader(train_data, batch_size=train_loader.batch_size, collate_fn=train_loader.collate_fn,
                                  num_workers=train_loader.num_workers)
        val_loader = DataLoader(val_data, batch_size=train_loader.batch_size, collate_fn=train_loader.collate_fn,
                                num_workers=train_loader.num_workers)

        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for waveform, labels in train_loader:
                waveform, labels = waveform.to(device), labels.to(device)
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(1)

                with autocast():
                    outputs = self.model(waveform)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if early_stopping(avg_val_loss):
                break

        torch.save(self.model.state_dict(), f"models/{model_name}.pth")

    def validate(self, val_loader: DataLoader)-> float:
        """
         Validates the LEAN model on a validation DataLoader.

         Parameters:
         - val_loader (DataLoader): DataLoader for validation data.

         Returns:
         - float: Average validation loss.
         """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for waveform, labels in val_loader:
                waveform, labels = waveform.to(device), labels.to(device)
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(1)

                outputs = self.model(waveform)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def eval(self, test_dataloader: DataLoader, channels:int)-> (str, dict):
        """
        Evaluates the LEAN model on a test DataLoader.

        Parameters:
        - test_dataloader (DataLoader): DataLoader for test data.

        Returns:
        - tuple: A tuple containing the accuracy and classification report.
        """
        self.model.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for waveform, labels in test_dataloader:
                waveform, labels = waveform.to(device), labels.to(device)
                if waveform.dim() == 2:  # If missing channel dimension
                    waveform = waveform.unsqueeze(1)  # [batch, features] → [batch, 1, features]

                outputs = self.model(waveform)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(all_labels, all_preds, output_dict=True)
        print(report)
        return accuracy, report
