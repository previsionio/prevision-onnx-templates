# prevision-onnx-templates

This repo provide code to understand and build onnx models for use in the [Prevision.io Platform](https://cloud.prevision.io).


A complete doc for using onnx model in [Prevision.io Platform](https://cloud.prevision.io) is available [here](https://previsionio.readthedocs.io/fr/latest/studio/experiments/external-model.html)

## Setup

Clone this repo and open the prevision-onnx-templates folder :

```sh
git clone https://github.com/previsionio/prevision-onnx-templates.git
cd prevision-onnx-templates
```


If not done, create a virtualenv 

```sh
python -m venv env
```

And install the requirements 

``` 
source env/bin/activate
pip install -r requirements.txt
```

### Using the SDK

If you want to use the [Prevision SDK](https://prevision-python.readthedocs.io/en/latest/source/getting_started.html) yoy may setup an `.env` file with the following fields :

```
DOMAIN=https://cloud.prevision.io
PIO_MASTER_TOKEN=<MASTER_TOKEN>
``` 

( see [the API doc to get your master token](https://previsionio.readthedocs.io/fr/latest/API/using.html))

## Usage

Note : a small subset of a fraud transaction dataset is provided in the `data` folder.

You can use sktoonnx.py for a basic sklearn classifier exportation to onnx :

```sh
python sktoonnx.py
``` 

The scripts build a ( very ) basic classifier and export it to onnx model. You can then use the content of `data/assets` folder into your [Prevision Account](https://cloud.prevision.io). Then see [the documentation for importing your own model](https://previsionio.readthedocs.io/fr/latest/studio/experiments/external-model.html)


### Using the SDK

You can use ( and inspect )  the `sdkimport.py` script to export your models and pipeline programmatically. This script build a basic sklearn pipeline ( feature engineering + modelisation) , convert it to onnx and import it in your [prevision account](https://cloud.prevision.io)


```sh
python sdkimport.py
``` 
