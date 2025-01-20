# 🌳🪓 Detecting illegal deforestation

## Description

## Data

The dataset is publicaly available at HuggingFace with an account.  The chainsaw dataset was compiled to train a model that could run on devices in the forest and detect illegal logging in real-time. The devices send a message to rangers on the ground to make an intervention.

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

`pip install librosa soundfile datasets`

#### Load data:

``` python
from datasets import load_dataset
dataset = load_dataset("rfcx/frugalai", streaming=True)
print(next(iter(dataset['train'])))
```

## Timeline