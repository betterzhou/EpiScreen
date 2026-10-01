# EpiScreen


## 1. Introduction
This repository contains code for the paper "Early Epilepsy Detection from Electronic Health Records with Large Language Models" (npj Digital Medicine 2026).


## 2. Usage


Training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --mode train

Test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --mode test --output_file ./results/test_results.xlsx

### Environment

Install packages:


### Datasets




## 3. Contact

For research cooperation, please contact shuang DOT zhou AT connect.polyu DOT hk



## 4. Citation
Please kindly cite the paper if you are interested in our work.
```bib
@article{zhou2026episcreen,
  title={EpiScreen: Early Epilepsy Detection from Electronic Health Records with Large Language Models},
  author={Zhou, Shuang and Yu, Kai and Zhan, Zaifu and Zhou, Huixue and Zeng, Min and Xie, Feng and Sha, Zhiyi and Zhang, Rui},
  journal={arXiv preprint arXiv:2603.28698},
  year={2026}
}
```

