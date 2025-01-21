import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def model_eval(model, test_dataloader):
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_dataloader:
            batch_features = batch_features.view(batch_features.size(0), -1)
            outputs = model(batch_features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "tiny_dnn_nature_chainsaw.pth")