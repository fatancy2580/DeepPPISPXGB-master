# **DeepPPISPXGB** 
A Deep Learning and XGBoost-based Method for Predicting Protein-protein Interaction Sites 


# **installation** 

umap-learn==0.4.6

numpy==1.15.0

scikit-learn==0.23.2

xgboost==1.1.0


# **data**

local_feature_training_set.csv: Preprocessing data of feature extractor contains 65869 rows and 344 columns data. The first 343 columns represent feature and the last column represent label (The CSV file contains row indexes and column index) 

local_feature_testing_set.csv: Preprocessing data of feature extractor contains 11791 rows and 344 columns data. The first 343 columns represent feature and the last column represent label (The CSV file contains row indexes and  column index)

global&local_feature_training_set.csv: Preprocessing data of feature extractor contains 65869 rows and 1028 columns data. The first 1027 columns represent feature and the last column represent label (The CSV file contains row indexes and  column index)

global&local_feature_testing_set.csv: Preprocessing data of feature extractor contains 11791 rows and 1028 columns data. The first 1027 columns represent feature and the last column represent label(The CSV file contains row indexes and  column index) 

raw_feature_training_set.csv: raw feature data (secondary structure, raw protein sequence, position specific scoring matrix feature)  contains 65869 rows and 24844 columns data. The first 24843 columns represent feature and the last column represent labell(The CSV file contains  column index).

raw_feature_testing_set.csv: raw feature data (secondary structure, raw protein sequence, position specific scoring matrix feature)  contains 11791 rows and 24844 columns data. The first 24843 columns represent feature and the last column represent label(The CSV file contains  column index).



URL: https://data.mendeley.com/drafts/9tft3vz5tm

# **Usage** 

The knowledge about protein-protein interactions is beneficial to understand cellular mechanism. Protein-protein interactions are usually determined according to their protein-protein interaction sites. Due to limitation of current techniques, it is still a challenging task to detect protein-protein interaction sites. In this article, we presented a deep learning and XGBoost-based method (called DeepPPISP-XGB) for predicting protein-protein interaction sites. The deep learning model served as feature extractor to remove redundant information of the protein sequences. The Extreme Gradient Boosting algorithm was used to construct a classifier for predicting protein-protein interaction sites. The DeepPPISP-XGB achieved area under the receiver operating characteristic curve of 0.681, a recall of 0.624 and area under the precision-recall curve of 0.339, being competitive with the state-of-the-art methods. We also validate the positive role of global features in predicting protein-protein interaction sites. 

# **The architecture of DeepPPISPXGB** 
[修改流程图.pptx](https://github.com/fatancy2580/DeepPPISPXGB-master/files/7119663/default.pptx)




