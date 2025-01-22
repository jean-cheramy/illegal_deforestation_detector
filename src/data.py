import librosa
import numpy as np
import torch
from torch.utils.data import IterableDataset


class AudioDataset(IterableDataset):
    def __init__(self, dataset, max_length=16000, feature_length=50):
        self.dataset = dataset
        self.max_length = max_length
        self.feature_length = feature_length

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for sample in self.dataset:
            audio = sample['audio']
            y = audio['array']
            sr = audio['sampling_rate']
            label = sample['label']

            # Pad or truncate audio to max_length
            if len(y) > self.max_length:
                y = y[:self.max_length]
            else:
                y = np.pad(y, (0, self.max_length - len(y)), 'constant')

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Extract log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)

            # Combine features
            features = np.concatenate([mfcc, log_mel_spec], axis=0)

            # Pad or truncate features to fixed length
            if features.shape[1] > self.feature_length:
                features = features[:, :self.feature_length]
            else:
                features = np.pad(features, ((0, 0), (0, self.feature_length - features.shape[1])), 'constant')

            # Yield features and label
            yield torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
