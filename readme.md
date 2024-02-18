
# Introduction
Paper:
```
MKG-GC: a multi-task learning-based knowledge graph framework with personalized application to gastric cancer
```

The project includes code for entity extraction, entity normalization, relational classification, and knowledge embedding. 

# Dataset

Download link for the dataset： 
- baidu drive: [link](https://pan.baidu.com/s/1MFi3Pn9-9smapogtMOdx7w)  code: ezvp
- google drive：[link](https://drive.google.com/file/d/1jkT8ZhLW1GP6SVdOF11JQ3AnDUqJM0aN/view?usp=sharing)

Note： Due to the limitations of the agreement between UMLS and DrugBank, we are unable to provide corresponding data

## entity extraction dataformat

BIO format
for example:
```
gastric B-Disease
cancer I-Disease
is O
a O
malignant B-Disease
disease I-Disease
. O
```
## entity normalization dataformat
Each medical entity corresponds to an ID, and multiple medical entity names have the same ID
For Example:
```
gastric cancer|D013274
stomach neoplasm|D013274
cancer|D009369
```

## relation classification dataformat
Two entities of each piece of data are surrounded by special tokens
Text|entity1|entity2|entity_type1|entity_type2
```
We found that the expression of all these genes, including thymidylate synthase (TS),  [s2] dihydrofolate reductase [e2]  (DHFR), ribonucleotide reductase (PNR), proliferating cell nuclear antigen (PCNA),  [s1] histone [e1]  H1, histone H2A + 2B, histone H3, and histone H4, was induced to high levels in young IMR-90 cells but not in old IMR-90 cells.	histone	dihydrofolate reductase	protein	protein	
```




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
For help or issues using MKG-GC, please submit a GitHub issue. 

