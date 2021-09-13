# Towards Automatic Bias Detection in Knowledge Graphs
Repository for short paper accepted in EMNLP 2021 Findings, you may find our paper here

## Data
### FB15K-237
You may download our trained models from [here](https://polybox.ethz.ch/index.php/s/pLp8Bmp9abrytIQ) in directory `trained_models/`, and uncompress it.

### Wikidata 5M
Get Wikidata5m Pre-trained embeddings (TransE, DistMult, ComplEx, RotatE) from [here](https://graphvite.io/docs/latest/pretrained_model.html), and put inside the directory `data/wiki5m`. Since we only work around human-related triples, we filtered and saved needed entities and relations as `human_ent_rel_sorted_list.pkl` in directory `data/wiki5m`. 

Run the following commands to first save human-relate embeddings, and then wrap into its corresponding pykeen trained model which will be saved in the directory `trained_models/wiki5m`
```
python process_wiki5m.py
mkdir -p trained_models/wiki5m
python wrap_wiki5m.py
```
## Classification (\tail prediction)
To classify the entities according to the target relation, please refer to the code in experiments/run_tail_prediction.py
In the paper as well as the code files, the target relation is profession - meaning that we train a classifier on the task of predicting the profession for each entity. 
