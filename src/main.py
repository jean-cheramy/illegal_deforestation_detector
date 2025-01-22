from codecarbon import track_emissions
from datasets import load_dataset
from torch.utils.data import DataLoader

from architectures.LEAN import train_LEAN
from data import AudioDataset
from utils import collate_fn


@track_emissions(log_level="critical")
def main():
    dataset = load_dataset("rfcx/frugalai")
    train_dataset = dataset["train"].select(range(1000))
    test_dataset = dataset["test"].select(range(100))

    max_length = 16000
    feature_length = 50
    train_audio_dataset = AudioDataset(train_dataset, max_length=max_length, feature_length=feature_length)
    test_audio_dataset = AudioDataset(test_dataset, max_length=max_length, feature_length=feature_length)
    batch_size = 32
    num_epochs = 10

    train_dataloader = DataLoader(train_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)

    train_LEAN(train_dataloader, test_dataloader, num_epochs)


if __name__ == "__main__":
    main()