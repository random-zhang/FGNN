# FGNN

## Introduction
### A new fingerprint and graph hybrid neural network for predicting molecular properties

## Model Structure
![Model Structure](https://foruda.gitee.com/images/1718702515165197457/9d691265_7602785.png "屏幕截图")  
## Installation  
#### To install the project, follow these steps:  
1.create conda env 
`conda create --name fgnn python=3.8`  
`conda activate fgnn`  
2.install dependencies  
`pip install torch==1.11.6+cu116`  
`pip install torchvision==0.14.1+cu116`  
`pip install mrmr-selection==0.2.8`  
`pip install numpy==1.23.3`  
`pip install optuna==3.3.0`  
`pip install pandas==2.0.3`  
`pip install tqdm==4.61.2`   
`pip install shap==0.44.0`   
`pip install scikit-learn==1.3.1`   
`pip install scipy==1.10.1`   
`pip install tensorboard==2.10.1`   
`pip install rdkit-pypi`  
`pip install matplotlib==3.6.1`


 



## Usage
### train and validation   
`python train.py --task_name xxx --split_type xxx --fp_dim xxxx --select_method xxx`  
### for example 
`python train.py --task_name bbbp --split_type random --fp_dim 2300 --select_method selected`  

### predict   
`python predict.py --task_name xxx --split_type xxx --fp_dim xxxx --select_method xxx`




