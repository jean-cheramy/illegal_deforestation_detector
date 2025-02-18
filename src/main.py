import inspect
import json
import warnings
from typing import Dict, Any

from datasets import load_dataset
from torch.utils.data import DataLoader

from architectures.lean import LEAN
from architectures.linear import Linear
from architectures.rf import RandomForest
from architectures.svm import SVM
from src.utils.dataloader import AudioDataset
from src.utils.emissions import tracker, clean_emissions_data
from src.utils.utils import collate_fn

warnings.filterwarnings("ignore")


def format_results(emissions_data: Any, accuracy: float, report: dict) -> Dict[str, Any]:
    """
    Formats the results of a model training run.

    Parameters:
    - emissions_data: Emissions data from the tracker
    - accuracy: Accuracy score
    - report: Classification report

    Returns:
    - Dictionary containing formatted results
    """
    return {
        "emission": clean_emissions_data(emissions_data),
        "energy_consumed_wh": emissions_data.energy_consumed * 1000,
        "emissions_gco2eq": emissions_data.emissions * 1000,
        "metrics": {"accuracy": accuracy, "report": report}
    }


def generic_start_training(model: Any, train_dataloader: DataLoader, test_dataloader: DataLoader, model_name: str, **kwargs)-> dict:
    """
    Generic function to train and evaluate models.

    Parameters:
    - model_class: Model class to instantiate (e.g., LEAN, Linear, RandomForest, SVM)
    - model_name: Name of the model for emissions tracking
    - train_dataloader: DataLoader for training
    - test_dataloader: DataLoader for testing
    - kwargs: Additional arguments like epochs, channels, etc.

    Returns:
    - Dictionary with emissions data and evaluation metrics
    """

    tracker.start()
    tracker.start_task(model_name)

    # Check if the model's __init__ requires specific arguments (e.g., 'channels')
    init_params = dict(inspect.signature(model.__init__).parameters)
    init_params.pop("self", None)

    model_args = {key: kwargs[key] for key in init_params if key in kwargs}
    train_args = {key: kwargs[key] for key in kwargs if key not in init_params}
    eval_args = {key: kwargs[key] for key in ['channels'] if key in kwargs}

    # Create model instance with required args
    model = model(**model_args)

    # Train the model, passing only training-specific arguments (no 'channels' here)
    model.train(train_dataloader, model_name, **train_args)

    # Evaluate the model, passing only the necessary parameters
    accuracy, report = model.eval(test_dataloader, **eval_args)
    emissions_data = tracker.stop_task()
    return format_results(emissions_data, accuracy, report)


def main():
    """
    Main function to train and evaluate different models with and without chainsaw feature reduction.

    This function loads the dataset, initializes the data loaders, trains each model, and saves the results to a JSON file.
    """
    results_dict = {}
    dataset = load_dataset("rfcx/frugalai")
    train_dataset = dataset["train"].select(range(1000))
    test_dataset = dataset["test"].select(range(200))
    batch_size = 128
    epochs = 1

    # Model training without chainsaw feature reduction
    train_audio_dataset = AudioDataset(train_dataset)
    test_audio_dataset = AudioDataset(test_dataset)

    train_dataloader = DataLoader(train_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)
    test_dataloader = DataLoader(test_audio_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)

    models = [
        (Linear, "linear_regression"),
        (LEAN, "lean", {"channels": 48, "epochs": epochs}),
        (RandomForest, "random_forest"),
        (SVM, "svm")
    ]

    for model_entry in models:
        if len(model_entry) == 3:
            model_class, model_name, extra_args = model_entry
        else:
            model_class, model_name = model_entry
            extra_args = {}  # Default empty dictionary

        print(f"Training {model_name}...")
        results_dict[model_name] = generic_start_training(
            model_class, train_dataloader, test_dataloader, model_name, **extra_args
        )
        with open("PerformanceVSEfficiency.json", "w+", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

    # Model training with chainsaw feature reduction

    train_audio_dataset_chainsaw = AudioDataset(train_dataset, chainsaw=True)
    test_audio_dataset_chainsaw = AudioDataset(test_dataset, chainsaw=True)

    train_dataloader_chainsaw = DataLoader(train_audio_dataset_chainsaw, batch_size=batch_size, collate_fn=collate_fn,
                                           num_workers=1)
    test_dataloader_chainsaw = DataLoader(test_audio_dataset_chainsaw, batch_size=batch_size, collate_fn=collate_fn,
                                          num_workers=1)

    models_chainsaw = [
        (Linear, "linear_regression_chainsaw"),
        (LEAN, "lean_chainsaw", {"channels": 1, "epochs": epochs}),
        (RandomForest, "random_forest_chainsaw"),
        (SVM, "svm_chainsaw")
    ]

    for model_entry in models_chainsaw:
        if len(model_entry) == 3:
            model_class, model_name, extra_args = model_entry
        else:
            model_class, model_name = model_entry
            extra_args = {}

        print(f"Training {model_name}...")
        results_dict[model_name] = generic_start_training(
            model_class, train_dataloader_chainsaw, test_dataloader_chainsaw, model_name, **extra_args
        )
        with open("PerformanceVSEfficiency.json", "w+", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

    with open("PerformanceVSEfficiency.json", "w+", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
