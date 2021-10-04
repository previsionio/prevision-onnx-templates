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

## Usage

Note : a small subset of a fraud transaction dataset is provided in the `data` folder.

You can use sklearn_classif_to_onnx.py for a basic sklearn classifier exportation to onnx :

```sh
python sktoonnx.py
``` 

The scripts build a ( very ) basic classifier and export it to onnx model. You can then use the content of `data/assets` folder into your [Prevision Account](https://cloud.prevision.io). 