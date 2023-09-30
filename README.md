# BELS: Datasets and Codes
A Broad Ensemble Learning System for Drifting Stream Classification

In a data stream environment, classification models must effectively and efficiently handle
concept drift. Ensemble methods are widely used for this purpose; however, the ones available in the
literature either use a large data chunk to update the model or learn the data one by one. In the former,
the model may miss the changes in the data distribution, while in the latter, the model may suffer from
inefficiency and instability. To address these issues, we introduce a novel ensemble approach based on the
Broad Learning System (BLS), where mini chunks are used at each update. BLS is an effective lightweight
neural architecture recently developed for incremental learning. Although it is fast, it requires huge data
chunks for effective updates and is unable to handle dynamic changes observed in data streams. Our
proposed approach, named Broad Ensemble Learning System (BELS), uses a novel updating method that
significantly improves best-in-class model accuracy. It employs an ensemble of output layers to address
the limitations of BLS and handle drifts. Our model tracks the changes in the accuracy of the ensemble
components and reacts to these changes. We present the mathematical derivation of BELS, perform
comprehensive experiments with 35 datasets that demonstrate the adaptability of our model to various
drift types, and provide its hyperparameter, ablation, and imbalanced dataset performance analysis. The
experimental results show that the proposed approach outperforms 10 state-of-the-art baselines, and supplies
an overall improvement of 18.59% in terms of average prequential accuracy.

# Datasets
All the real and synthetic datasets are available in two formats : CSV and ARFF. 

Google Drive Link: https://drive.google.com/drive/folders/16Nn9xmaMjbrR1btzN7__sSA5LXEgZyQc?usp=sharing

# Requirements
Python: 3.10.9 <br />
Numpy: 1.23.5 <br />
Pandas:  1.5.3 <br />

# Running BELS

To execute the code, ensure that all the code files and the dataset (in .CSV format) are placed within the same folder. In the BELS_test.py file, make sure to include your dataset name using the "dataset_name" variable in the format: dataset_name = "YOUR_DATASET_NAME". After making this change, run the BELS_test.py file.

# Citing BELS

```plaintext
@ARTICLE{10225305,
  author={Bakhshi, Sepehr and Ghahramanian, Pouya and Bonab, Hamed and Can, Fazli},
  journal={IEEE Access}, 
  title={A Broad Ensemble Learning System for Drifting Stream Classification}, 
  year={2023},
  volume={11},
  number={},
  pages={89315-89330},
  doi={10.1109/ACCESS.2023.3306957}}
```



