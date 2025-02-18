import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Tuple
from src.utils.extract_chainsaw import extract_chainsaw_features


class AudioDataset(Dataset):
    """Dataset class for audio data.

      Handles feature extraction and formatting for audio samples.
      """
    def __init__(self, dataset: Any, chainsaw: bool=False, max_length: int=1000, feature_length: int=1000):
        self.dataset = dataset
        self.max_length = max_length
        self.feature_length = feature_length
        self.extract_chainsaw_only = chainsaw

    def format_features(self, features: np.ndarray, label: int)-> Tuple[torch.Tensor, torch.Tensor]:
        """Formats features and label into tensors.

        Pads or truncates features to a fixed length and converts them to a PyTorch tensor.

        Args:
            features (np.ndarray): NumPy array of audio features.
            label (int): Label of the audio sample.

        Returns:
            Tuple containing the formatted features tensor and label tensor.
        """
        # Pad or truncate features to fixed length
        if features.ndim == 1:
            if features.shape[0] > self.feature_length:
                features = features[:self.feature_length]
            else:
                features = np.pad(features, (0, self.feature_length - features.shape[0]), 'constant')
        elif features.ndim == 2:
            if features.shape[1] > self.feature_length:
                features = features[:, :self.feature_length]
            else:
                features = np.pad(features, ((0, 0), (0, self.feature_length - features.shape[1])), 'constant')

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the item at the given index.

        Retrieves the audio sample and label, extracts features, and formats them into tensors.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple containing the features tensor and label tensor.
        """
        sample = self.dataset[idx]
        audio = sample['audio']
        y = audio['array']
        sr = audio['sampling_rate']
        label = sample['label']

        if not self.extract_chainsaw_only:
            # Pad or truncate audio to max_length
            if len(y) > self.max_length:
                y = y[:self.max_length]
            else:
                y = np.pad(y, (0, self.max_length - len(y)), 'constant')

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)

            # Extract log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            log_mel_spec = librosa.power_to_db(mel_spec)

            # Combine features
            features = np.concatenate([mfcc, log_mel_spec], axis=0)
            return self.format_features(features, label)
        else:
            try:
                return self.format_features(extract_chainsaw_features(y, sr=sr), label)
            except Exception as e:
                print(f"An error occurred while extracting chainsaw features: {e}")
                return torch.empty(0), torch.empty(0, dtype=torch.long)

