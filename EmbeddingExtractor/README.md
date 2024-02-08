# ALIGNNTL: Feature Extraction

This directory contains information on how to perform feature extraction using ALIGNN.

### Instructions

The user requires following files in order to perform feature extraction
* Sturcture files - contains structure information for a given material (format: `POSCAR`, `.cif`, `.xyz` or `.pdb`) 
* Input-Property file - contains name of the structure file and its corresponding property value (format: `.csv`)
* Pre-trained model - model trained using ALIGNN using any specific materials property (format: `.zip`)

We have provided the an example of Sturcture files (`POSCAR` files) and Input-Property file (`id_prop.csv`) in [`examples`](../examples). Download the pre-trained model trained on large datasets from <a href="https://figshare.com/projects/ALIGNN_models/126478">here</a>. 

Now, in order to perform feature extraction, add the details regarding the model in the `all_models` dictionary inside the `train.py` file as described below:
```
all_models = {
    name of the file: [link to the pre-trained model (optional), number of outputs],
    name of the file 2: [link to the pre-trained model 2 (optional), number of outputs],
    ...
    }
```
If the link to the pre-trained model is not provided inside the `all_models` dictionary, place the zip file of the pre-trained model inside the [`alignn`](./alignn) folder. Once the setup for the pre-trained model is done, the feature extraction can be performed by running the `create_features.sh` script file which contains the following code:
```
for filename in ../examples/*.vasp; do
    python alignn/pretrained_activation.py --model_name mp_e_form_alignnn --file_format poscar --file_path "$filename" --output_path "../examples/data"
done
```
The script will convert the structure files into atom (x), bond (y) and angle (z) based features one-by-one (batch-wise conversion has not been implemented yet).  Example: `abc.vasp` will produce `abc_x.csv` (9 atom-based features), `abc_y.csv` (9 bond-based features) and `abc_z.csv` (5 angle-based features). 

Once you have converted all the structure files in the Input-Property file `id_prop.csv` using the script file, run the jupyter notebooks `pre-processing.ipynb` to convert the structure-wise features into a dataset. Pre-processing steps contained within the `pre-processing.ipynb` file is as follows:
* Attach the appropriate property value and identifier (jid) to each of the extracted features file based on id_prop.csv 
* Create a seperate file for each of the features (atom, bond, angle) based on the extracted checkpoints
* Create combined features (in the order of atom, bond and angle) from same (3-1) or different (3-2) checkpoints. Use first 512 features for atom+bond and all features for atom+bon+angle as input for model training.
* (Optional) Divide each of the files into train, validation and test files based on the json file `ids_train_val_test.json` available in the output directory of the ALIGNN model
