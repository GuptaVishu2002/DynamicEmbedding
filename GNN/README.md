# DynamicEmbedding: GNN

This directory contains information on how to perform model training using dynamic embedding with Roost.

### Instructions

The user requires following files in order to start training a model
* Embedding files - contains embedding informations for a given material (format: `.csv`) 
* Input-Property file - contains files with materials-ids, composition strings and target values as the columns. (format: `.csv`)

We have provided the an example of Input-Property file (`id_prop.csv`) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>. 
 
Once the setup is done, model training can be performed as follows:
```
python examples/roost-example.py --fea-path data/embeddings/256-embedding.json --epochs 250 --train --evaluate --data-path data/id_prop/sample_train_set.csv --val-path data/id_prop/sample_val_set.csv --test-path data/id_prop/sample_test_set.csv --fea_num 9
```
Make sure that the files are correctly places in the correct directories. 

