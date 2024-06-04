# Optimizing for ROC Curves on Class-Imbalanced Data by Training over a Family of Loss Functions

Official code implementation for ["Optimizing for ROC Curves on Class-Imbalanced Data by Training over a Family of Loss Functions".](https://arxiv.org/abs/2402.05400)

## Getting started
- Details about the conda environment are specified in `environment.yml`.
- Details about how to configure Kaggle datasets are in `configure_kaggle_datasets.ipynb`.

## Repository structure
- `train.py` contains code to train both models with VS loss and VS loss + LCT models and `test.py` contains code to test these models. Both `train.py` and `test.py` are based on the Trainer specified in `trainer.py`.
- Arguments for training and testing models are ing `args`. General arguments, such as data configurations, model architecture, and loss hyperparamters can be found in ``base_args.py``. More specific training arguments, such as learning rates and batch sizes, are in ``train_args.py`` and more specific testing arguments, such as the $\mathbf{\lambda}$ values to use for evaluation, are in `test_args.py`.
- `configs` contains configurations for each dataset used in the paper. Specifically, these configs contain information about how to subset base datasets (e.g., CIFAR10) to binary forms as well as training configurations for the Kaggle datasets.
- `data` contains information about each base dataset (e.g., transforms, loaders, and label maps).
- `models` contains the model architectures used, including model architectures with FiLM layers for LCT.
- `losses` contains code for VS loss and VS loss + LCT for any $\mathbf{\lambda}$.
- `optimizers` contains code for the Sharpness Aware Minimization (SAM) optimizer.
- All information about runs, including checkpoints and results, will be saved in a directory within the `runs` directory which will be automatically named based on the given arguments.

## Example commands
### Baseline
This command trains a model on the cat/dog subset of CIFAR10 with $\beta=100$. This model is trained with VS loss (baseline method) with $\Omega=0.5, \gamma=0.2, \tau=2$.
```
python train.py configs/cifar10_cat_dog.yml --beta 100 --model resnet32 --omega 0.5 --gamma 0.2 --tau 2
```
### LCT $\mathbf{\lambda}=\tau$
This command trains a model on the same dataset, but with Loss Conditional Training (LCT) applied to the $\tau$ hyperparameter in VS loss. In other words, $\tau$ is drawn from a distribution with each batch and used as additional input to the model. We specify this distribution to be a linear distribution with range $[a,b]=[0,3]$ and the height at $b$, $h_b=0.33$. $\Omega, \gamma$ are set to constants as in the original VS loss formulation. resnet32lct_penultimate is the model architecture which includes FiLM layers for the additional $\lambda$ input.
```
python train.py configs/cifar10_cat_dog.yml --beta 100 --model resnet32lct_penultimate --omega 0.5 --gamma 0.2 --tau 0 3.0 0.33
```
### LCT $\mathbf{\lambda}=(\Omega, \gamma, \tau)$
This command trains a model on the same dataset, but with LCT with $\mathbf{\lambda}=(\Omega, \gamma, \tau)$. In other words, $\Omega, \gamma, \tau$ are each drawn from a distribution with each batch and used as additional input to the model. We specify this distribution separately for each hyperparameter. 
```
python train.py configs/cifar10_cat_dog.yml --beta 100 --model resnet32lct_penultimate --omega 0.5 1 2 --gamma 0.0 0.4 2.5 --tau 0 3.0 0.33
```
### Baseline + SAM
This command trains a model equivalent to the first example, but with Sharpness Aware Minimization (SAM) with $\rho=0.3$ instead of SGD.
```
python train.py configs/cifar10_cat_dog.yml --beta 100 --omega 0.5 --gamma 0.2 --tau 2 --optimizer sam --sam_rho 0.3
```
### Baseline on an arbitrary CIFAR10 pair
This command trains a model equivalent to the first example, but on a different subset of CIFAR10 (i.e., airplane (label 0) vs. bird (label 2)).
```
python train.py --cifar10_pair 0 2 --beta 100 --omega 0.5 --gamma 0.2 --tau 2
```
### Finetuning with LCT $\mathbf{\lambda}=\tau$ on SIIM-ISIC Melanoma dataset 
This command finetunes a resNext50-32x4d model pretrained on ImageNet with the SIIM-ISIC Melanoma dataset using LCT $\mathbf{\lambda}=\tau$.
```
python train.py configs/melanoma.yml --beta 100 --model resNext50-32x4d_lct --omega 0.5 --gamma 0.2 --tau 0.0 3.0 0.33
```
### Testing models
To test models trained with the previous commands, simply use the same arguments with `test.py` instead of `train.py`.


## Citation and License
If you use our work, please cite:
```
@misc{lieberman2024optimizing,
      title={Optimizing for ROC Curves on Class-Imbalanced Data by Training over a Family of Loss Functions}, 
      author={Kelsey Lieberman and Shuai Yuan and Swarna Kamlam Ravindran and Carlo Tomasi},
      year={2024},
      eprint={2402.05400},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` 