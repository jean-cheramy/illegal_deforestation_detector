import torch

def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    return features, labels
