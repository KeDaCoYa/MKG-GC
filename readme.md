
# Introduction
Paper:
```
GCKG: a multi-task learning-based framework for Gastric Cancer Knowledge Graph and drug discovery
```

The project includes code for entity extraction, entity normalization, relational classification, and knowledge embedding. 

# Dataset

Download link for the dataset： https://pan.baidu.com/s/1MFi3Pn9-9smapogtMOdx7w  code: ezvp


Note： Due to the limitations of the agreement between UMLS and DrugBank, we are unable to provide corresponding data

## entity extraction dataformat
``

``
## entity normalization dataformat

## relation classification dataformat




# Run

You need to download biobert first: https://github.com/dmis-lab/biobert

## entity extraction

```bash
cd entity_extraction
bash multi_run.sh
```

## entity extraction

```bash
cd entity_extraction
bash multi_run.sh
```
## entity normalization

```bash
cd entity_normalization
bash multi_train.sh
```

## relation classification

```bash
cd relation_classification
bash multi_run.sh
```



# requirements
```
python>=3.8
transformer == 3.0.0
pytorch >= 1.7
```

# Contact Information
For help or issues using GCKG, please submit a GitHub issue. 

