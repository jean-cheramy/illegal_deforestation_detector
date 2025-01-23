from codecarbon import track_emissions
from datasets import load_dataset
from torch.utils.data import DataLoader

from architectures.LEAN import train_LEAN
from architectures.RF import RandomForest
from data import AudioDataset
from utils import collate_fn


@track_emissions(log_level="critical")
def main():
    dataset = load_dataset("rfcx/frugalai")
    train_dataset = dataset["train"]#.select(range(1000))
    test_dataset = dataset["test"]#.select(range(200))

    max_length = 48000
    feature_length = 100
    train_audio_dataset = AudioDataset(train_dataset, max_length=max_length, feature_length=feature_length)
    test_audio_dataset = AudioDataset(test_dataset, max_length=max_length, feature_length=feature_length)
    batch_size = 128
    num_epochs = 10

    train_dataloader = DataLoader(train_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)
    test_dataloader = DataLoader(test_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)

    #train_LEAN(train_dataloader, test_dataloader, num_epochs)
    rf = RandomForest()
    rf.train(train_dataloader)
    rf.eval_model(test_dataloader)


if __name__ == "__main__":
    main()