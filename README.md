# Benefits of Permutation-Equivariance in Auction Mechanisms

---

This repo holds the code for the paper, _Benefits of Permutation-Equivariance in Auction Mechanisms_, which is accepted to NeurIPS 2022. 

## Getting Started 
We mainly build our code based on the [RegretNet](https://github.com/saisrivatsan/deep-opt-auctions). Thanks for their excellent work.

Before running our code, please install the following packages first:

- Python3
- Tensorflow2 (Note: It's OK to install Tensorflow v1, because we do not use any new feature for Tensorflow v2)
- Numpy
- Matplotlib
- Easydict 
- Git

---
## Training and Test 
To train and test RegretNet-PE, please switch the git branch to ```master``` first:

```shell
git checkout master
```

For RegretNet-test, use:
```shell
git checkout rtest
```

Changing branch would modified the network structure and pipeline. Then, the experiments can be conducted with:
```shell
python run_train.py [setting_name]
python run_test.py [setting_name]
```
All the ```setting_name``` are listed as followed

- additive_1x2_uniform
- additive_2x1_uniform
- additive_2x2_uniform
- additive_2x5_uniform
- additive_3x1_uniform
- additive_5x3_uniform
- additive_2x1_normal
- additive_2x2_normal
- additive_3x1_normal
- additive_5x3_normal
- additive_3x1_nor51
- additive_5x1_1010

---
