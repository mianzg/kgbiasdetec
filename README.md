# Towards Automatic Bias Detection in Knowledge Graphs
Repository for short paper accepted in EMNLP 2021 Findings, you may find our paper here

## Data
### FB15K-237
You may download our trained models from [here]() and put inside `trained_models/fb15k237`

### Wikidata 5M
Get Wikidata5m Pre-trained embeddings (TransE, DistMult, ComplEx, RotatE) from [here](https://graphvite.io/docs/latest/pretrained_model.html), and put inside the directory `data/wiki5m`. Since we only work around human-related triples, we filtered the needed entities and relations which you can download [here](https://polybox.ethz.ch/index.php/s/yCGuprCVAfK9aDl) and put inside `data/wiki5m`. 

Run the following commands to first save human-relate embeddings, and then wrap into its corresponding pykeen trained model which will be saved in the directory `trained_models/wiki5m`
```
python process_wiki5m.py
python wrap_wiki5m.py
```
