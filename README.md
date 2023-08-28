# DVSMN
This is the source code of DVSMN. This method was proposed in the paper **A Dual-View Style Mixing Network for Unsupervised Cross-Domain Fault Diagnosis With Imbalanced Data**.

![image](https://github.com/CQU-ZixuChen/DVSMN/blob/master/Framework.png)

The fundamental hyperparameters of DVSMN are set as follows:

  learning rate  ---  0.001
  
  **batch size     ---  64**
  
  training epoch ---  100

Operational environment：

  Python --- 3.9
  
  numpy  --- 1.20.0
  
  torch  --- 1.11.0+cu115
  
If this code is helpful to you, please cite this paper as follows, thank you!

@article{CHEN2023110918,
title = {A Dual-View Style Mixing Network for unsupervised cross-domain fault diagnosis with imbalanced data},
journal = {Knowledge-Based Systems},
volume = {278},
pages = {110918},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110918},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123006688},
author = {Zixu Chen and Wennian Yu and Liming Wang and Xiaoxi Ding and Wenbin Huang and Yimin Shao},
keywords = {Unsupervised cross-domain, Imbalanced data, Style mixing, Intermediate domain, Dual-view classifier},
abstract = {The remarkable progress of cross-domain fault diagnosis is based on the balanced distribution of different health conditions in a supervised manner. However, in engineering scenarios, the monitored fault data is scarce and imbalanced; variable working conditions and high labor costs make it luxurious to obtain labels; there is a huge gap between the current domain adaptation methods based on class balance data and real industrial applications. Therefore, a Dual-View Style Mixing Network (DVSMN) for dealing with unsupervised cross-domain fault diagnosis with imbalanced data is proposed. Two parallel graph convolution frameworks are first constructed to extract the fault features. Then, the style mixing module together with the domain style loss is proposed for obtaining generalized and domain-invariant representations without augmenting any synthetic samples. An intermediate domain can also be initialized to increase the original cross-domain overlap to facilitate the domain adaptation. Finally, a dual-view module that consists of a binary classifier and a multi-class classifier is constructed to realize sample-level dynamic re-weighting and accurate fault classification of imbalanced data. As such, the DVSMN can learn the generalized and domain-invariant features from the imbalanced data without any generative modules for sample re-balancing as well as target labels. Cross-domain experiments with different imbalance ratios are carried out via two datasets to validate the performance of the proposed method. Comparative studies with state-of-the-art methods and ablation experiments have demonstrated the effectiveness and superiority of the proposed method. The code of DVSMN is available at https://github.com/CQU-ZixuChen/DVSMN.}
}
