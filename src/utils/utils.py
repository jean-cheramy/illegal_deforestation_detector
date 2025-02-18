import torch
from torch.utils.data import DataLoader
import numpy as np

def collate_fn(batch: list):
    """Collates a list of (features, labels) tuples into a batched tensor.

      Handles empty and None values in the batch.

      Args:
          batch: A list of (features, labels) tuples.

      Returns:
          A tuple of batched features and labels tensors, or None if the batch is empty.
      """
    batch = [item for item in batch if item[0].numel() > 0]
    batch = [item for item in batch if item is not None]  # Remove None values
    if not batch:
        return None  # If all elements are None, return None to avoid breaking training
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    return features, labels


def prepare_data(dataloader: DataLoader)-> (np.ndarray, np.ndarray):
    """Prepares data for training or evaluation.

    Concatenates batches from a DataLoader, handles invalid data, and converts tensors to NumPy arrays.

    Args:
        dataloader (DataLoader): DataLoader containing the data.

    Returns:
        Tuple containing the concatenated data and labels as NumPy arrays, or (None, None) if an error occurs.
    """
    all_data = []
    all_labels = []

    for i, (batch_data, batch_labels) in enumerate(dataloader):
        try:
            # Ensure batch_data is a valid tensor
            if not isinstance(batch_data, torch.Tensor):
                print(f"Skipping batch {i}: batch_data is not a tensor")
                continue

            # Skip empty tensors or those with NaNs
            if batch_data.numel() == 0 or torch.isnan(batch_data).all():
                print(f"Skipping batch {i}: empty or all NaN values")
                continue

            all_data.append(batch_data)
            all_labels.append(batch_labels)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue  # Continue to the next batch even if an error occurs

    # Handle case where all batches were skipped
    if not all_data or not all_labels:
        print("No valid data found in dataloader.")
        return None, None

    # Concatenate all batches into single tensors
    try:
        concat_data = torch.cat(all_data, dim=0)
        concat_labels = torch.cat(all_labels, dim=0)
    except RuntimeError as e:
        print(f"RuntimeError during concatenation: {e}")
        return None, None

    # Flatten data if needed
    concat_data = concat_data.view(concat_data.size(0), -1)

    return concat_data.numpy(), concat_labels.numpy()
