# **DeepPPISPXGB** 
Protein-protein interaction site prediction Based on deep learning framework and XGBoost Algorithm

# **system requirement** 
DeepPPISPXGB is developed under Linux environment with python 3.7.5 Recommended RAM: > 24GB.  CUDA Version: 10.1 
# **installation** 

PyTorch==0.4.0

umap-learn==0.4.6

numpy==1.15.0

scikit-learn==0.23.2

xgboost==1.1.0

torch==1.3.0

# **Usage** 

The study of protein-protein interaction（PPIs）is helpful to understand cell function and develop drugs.However, conventional experimental methods of identifiing protein-protein interaction (PPI) sites is time-consuming and expensive.Therefore, many deep learning methods were used for predicting PPI sites.In this work, Sliding window is used for extracting local features,which can make full use of features of neighbors of a target amino acid.TextCNN was used for extracting the global features of the whole protein sequence. The deep learning model model trained firstly and the trained model served as feature extractor.The XGBoost model was used for predicting protein-protein interaction sites.The new deeping framework named DeepPPISPXGB.

# **The architecture of DeepPPISPXGB** 
![image](https://user-images.githubusercontent.com/52057084/119248317-eed63f80-bbc2-11eb-91a8-750f42eef47a.png)
# **The network structure of DeepPPISP** 
![image](https://user-images.githubusercontent.com/52057084/119248335-0f05fe80-bbc3-11eb-926a-36f4ca1899c1.png)

