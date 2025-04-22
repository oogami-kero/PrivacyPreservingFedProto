# FedProto: Federated Prototype Learning across Heterogeneous Clients (Research Fork)

**Note:** This repository is a fork of the original FedProto implementation [link to the original GitHub repository](https://github.com/yuetan031/FedProto) created for research purposes. The original paper can be found here: [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243).

---

## Purpose of this Fork

This fork is being developed as part of **Seunghoo Lee's Master's thesis research at National Taiwan University of Science and Technnology**.

The primary focus of this research is to:
1.  Investigate the privacy and security aspects of the FedProto federated learning algorithm.
2.  Analyze potential vulnerabilities associated with exchanging prototypes.
3.  Explore and potentially implement privacy-preserving techniques applicable to prototype-based federated learning.

Modifications and additions within this repository are made in service of these research goals.

---

*(The following sections are based on the original README and describe the base functionality)*

## Requirements
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
* **Important:** Datasets are NOT included in this repository and should be excluded via `.gitignore`.
* Download train and test datasets manually from the official sources (links below), or allow `torchvision` to download them automatically where applicable (e.g., using `download=True` in dataset constructors within the code).
* Original experiments were run on MNIST, FEMNIST, and CIFAR10.
    * MNIST: http://yann.lecun.com/exdb/mnist/
    * FEMNIST Data (derived from NIST SD19): https://s3.amazonaws.com/nist-srd/SD19/by_class.zip (Note: FEMNIST often requires specific preprocessing, refer to LEAF benchmark or original paper's methods if necessary)
    * CIFAR-10: http://www.cs.toronto.edu/~kriz/cifar.html

## Running the Base FedProto Experiments

* To train the FedProto on MNIST with n=3, k=100 under statistical heterogeneous setting:
```
python federated_main.py --mode task_heter --dataset mnist --num_classes 10 --num_users 20 --ways 3 --shots 100 --stdev 2 --rounds 100 --train_shots_max 110 --ld 1
```
* To train the FedProto on FEMNIST with n=4, k=100 under both statistical and model heterogeneous setting:
```
python federated_main.py --mode model_heter --dataset femnist --num_classes 62 --num_users 20 --ways 4 --shots 100 --stdev 2 --rounds 120 --train_shots_max 110 --ld 1
```
* To train the FedProto on CIFAR10 with n=5, k=100 under statistical heterogeneous setting:
```
python federated_main.py --mode task_heter --dataset cifar10 --num_classes 10 --num_users 20 --ways 5 --shots 100 --stdev 2 --rounds 110 --train_shots_max 110 --ld 0.1 --local_bs 32
```



You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'femnist', 'cifar10'
* ```--num_classes:```  Default: 10. Options: 10, 62, 10
* ```--mode:```     Default: 'task_heter'. Options: 'task_heter', 'model_heter'
* ```--seed:```     Random Seed. Default set to 1234.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--momentum:```       Learning rate set to 0.5 by default.
* ```--local_bs:```  Local batch size set to 4 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.


#### Federated Parameters
* ```--mode:```     Default: 'task_heter'. Options: 'task_heter', 'model_heter'
* ```--num_users:```Number of users. Default is 20.
* ```--ways:```      Average number of local classes. Default is 3.
* ```--shots:```      Average number of samples for each local class. Default is 100.
* ```--test_shots:```      Average number of test samples for each local class. Default is 15.
* ```--ld:```      Weight of proto loss. Default is 1.
* ```--stdev:```     Standard deviation. Default is 1.
* ```--train_ep:``` Number of local training epochs in each user. Default is 1.


## Citation
If you use the FedProto algorithm or base code from this repository (including this fork), please cite the original paper::
```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
