# **DeepPPISPXGB** 
Protein-protein interaction site prediction Based on deep learning framework and XGBoost Algorithm

# **system requirement** 
DeepPPISPXGB is developed under Linux environment with python 3.7.5 Recommended RAM: > 24GB. 
# **installation** 

PyTorch==0.4.0

umap-learn==0.4.6

numpy==1.15.0

scikit-learn==0.23.2

xgboost==1.1.0

torch==1.3.0

# **Usage** 

In this GitHub project, we give a demo to show how it works. 
The three benchmark datasets are given, i.e., Dset_186, Dset_72 and PDBset_164.
Dset_186 consists of 186 protein sequences with the resolution less than 3.0 Å with sequence homology less than 25%.
Dset_72 and PDBset_164 were constructed as the same as Dset_186. Dset_72 has 72 protein sequences and PDBset_164 consists of 164 protein sequences. 
These protein sequences in the three benchmark datasets have been annotated. 
Thus, we have 422 different annotated protein sequences. We remove two protein sequences for they do not have PSSM file.
The PSSMs, raw sequences, secondary structures and labels are given in data_cache folder. 
You can split the raw three datasets by yourself. In our study, we use the 83% as training dataset (350 protein sequences) and 17% as testing dataset (70 protein sequence). 
The detail of the three datasets and dataset division can see the paper and the code.

# **DeepPPISPXGB model architecture** 
![image](https://user-images.githubusercontent.com/52057084/113835819-e23a8b00-97be-11eb-8fa7-12984acf2db1.png)
