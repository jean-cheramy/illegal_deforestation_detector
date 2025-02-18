# 🌳🪓 Detecting illegal deforestation

## Description
This project is part of the [AI-Frugal Challenge](https://frugalaichallenge.org/). The challenge focuses on
developing resource-efficient AI models that achieve high performance while minimizing computational 
and energy costs.

## Data

The dataset is publicly available at HuggingFace with an account.  The chainsaw dataset was compiled to train a model that could run on devices in the forest and detect illegal logging in real-time. The devices send a message to rangers on the ground to make an intervention.

### Dataset Summary
A large set of short audio clips of chainsaws at varying distances. The data comes from Guardian devices deployed by Rainforest Connection to detect illegal logging. The majority of the recordings are from South America or South East Asia where Rainforest Connection has a large number of projects.

The dataset only contains audio data and labels. 

- Each audio clip is 3 seconds long. 
- Each sample is labelled either chainsaw (value 0 - positively identifying a chainsaw) or environment (value 1 - not containing a chainsaw).
- Train and test samples 

###  Sample
```json
{
    'audio': {
        'path': 'aoos_2021_02a16dd4-c788-4bbb-bc3d-e2f8322fe4b2_0-3.wav',
        'array': array([4.84344482e-01, 4.54193115e-01, 2.53906250e-02, ..., 2.44140625e-04, 3.05175781e-05, 9.15527344e-04]),
        'sampling_rate': 12000
    },
    'label': 0
}
```

### Getting started

#### Download files:
To download files from a gated dataset you’ll need to be authenticated. In the browser, this is automatic as long as you are logged in with your account. If you are using a script, you will need to provide a user token. In the Hugging Face Python ecosystem (transformers, diffusers, datasets, etc.), you can login your machine using the huggingface_hub library and running in your terminal:


`huggingface-cli login`

Alternatively, you can programmatically login using login() in a notebook or a script:

```python
from huggingface_hub import login
login()
```
#### Install requirements:

`pip install -r requirements.txt`

#### Load data:

``` python
from datasets import load_dataset
dataset = load_dataset("rfcx/frugalai", streaming=True)
print(next(iter(dataset['train'])))
```

## Methodology

### ML models

#### Linear Regression

- Chosen as a baseline model to understand simple relationships between extracted features and target outputs.
- Assumes a linear relationship, making it useful for benchmarking against more complex models.
- Provides interpretability but may struggle with non-linear patterns in audio data.
#### Random Forest

- A robust ensemble learning model that can handle non-linear relationships better than Linear Regression.
- Resistant to overfitting due to its decision tree structure and averaging mechanism.
- Works well with structured feature extraction techniques (e.g., spectral features).
#### Support Vector Machine (SVM)

- Effective for classification tasks if the goal is to distinguish chainsaw sounds from other noises.
- Handles high-dimensional feature spaces well, making it suitable for frequency-domain audio data.
- Works best with well-separated classes but may require careful tuning of kernel functions.
#### Convolutional Neural Network (CNN)

- Ideal for spectrogram-based audio analysis, capturing spatial patterns in frequency-time representations.
- Learns hierarchical features automatically, reducing the need for manual feature engineering.
- Can outperform traditional models if trained on sufficient data but requires more computational power.

By testing these models, we aim to balance simplicity, interpretability, and performance, selecting the best approach for extracting chainsaw-related audio features efficiently.

### Features engineering
To streamline our analysis, we've simplified the audio dataset by focusing on extracting chainsaw-related information from the first 1000 Hz of the frequency spectrum. Instead of analyzing the full range of sound frequencies, we apply a filter to isolate the lower-frequency components (0–1000 Hz), where key chainsaw characteristics are present.

This approach helps:

- Reduce complexity by eliminating unnecessary high-frequency data. 
- Enhance accuracy by focusing on the most relevant sound features. 
- Improve efficiency by minimizing the size of the dataset without losing crucial information.

By narrowing our analysis to this frequency range, we aim to maintain meaningful chainsaw-related patterns while simplifying data processing.
We will compare performance and carbon emission for both approaches, the exhaustive and the simplified.

### Results & Performance

 | Model                      | Accuracy | Carbon footprint |    |
|----------------------------|----------|------------------|-------|
| Linear regression          | Title    | Title            | Title |
| Random Forest              | Text     | Title            | Title |
| SVM                        | Title    | Title            | Title |
| CNN                        | Text     | Title            | Title |
| Linear regression chainsaw | Title    | Title            | Title |
| Random Forest chainsaw     | Text     | Title            | Title |
| Linear regression chainsaw | Title    | Title            | Title |
| Random Forest chainsaw             | Text     | Title            | Title |


## Usage 
The process can be triggered with this command line:

`python3 main.py`

## Team
[Vanessa Rivera-Quinones](https://github.com/vriveraq)  
[Antoine Gilliard](https://github.com/Tonyfunkman)  
[Jean Cheramy](https://github.com/jean-cheramy)

