import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.eval import model_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightweightAudioNet(nn.Module):
    def __init__(self):
        super(LightweightAudioNet, self).__init__()

        self.conv1 = nn.Conv1d(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

def train_LEAN(train_loader, test_loader, num_epochs):
    model = LightweightAudioNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    model.to(device)
    print("training...")
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for waveform, labels in train_loader:
            waveform, labels = waveform.to(device), labels.to(device)

            with autocast():
                outputs = model(waveform)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "lightweight_audio_model.pth")
#    model.load_state_dict(torch.load("lightweight_audio_model.pth"))
    model_eval(model, test_loader)
