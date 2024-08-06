# Introduction

This is the source code of our IJCAI 2024 paper "FineFMPL: Fine-grained Feature Mining Prompt Learning for Few-Shot Class Incremental Learning". Please cite the following paper if you use our code.


Hongbo Sun, Jiahuan Zhou, Xiangteng He, Jinglin Xu and Yuxin Peng, "FineFMPL: Fine-grained Feature Mining Prompt Learning for Few-Shot Class Incremental Learning", 33rd International Joint Conference on Artificial Intelligence (IJCAI), Jeju, South Korea, August 3-9, 2024.


# Dependencies

Python 3.7.15

PyTorch 1.13.0

Torchvision 0.14.0



# Data Preparation


The experiments are conducted on three standard few-shot class incremental learning datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please prepare the datasets following the guidelines in CEC (https://github.com/icoz69/CEC-CVPR2021). 




# Usage

Start training by executing the following commands. The experimental results will be saved in the ./output folder (if not exist, please create a blank folder first).

- CIFAR100

  ```
  bash train_cifar100.sh
  ```


- CUB200

  ```
  bash train_cub200.sh
  ```



- miniImageNet

  ```
  bash train_mini_imagenet.sh
  ```



For any questions, feel free to contact us (sunhongbo@pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.


## **Acknowledgement**


We appreciate the open-source contributions of the following work. Many thanks to the authors.

- [SAVC](https://github.com/zysong0113/SAVC)
- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [BiDistFSCIL](https://github.com/LinglanZhao/BiDistFSCIL?tab=readme-ov-file)