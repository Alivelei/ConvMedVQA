# Leveraging Convolutional Models as Backbone for Medical Visual Question Answering

## Install

1. Clone this repository and navigate to ConvMedVQA folder

   ```
   https://github.com/Alivelei/ConvMedVQA.git
   cd ConvMedVQA
   ```

2. Install Package: Create conda environment

   ```python
   conda create -n conv_medvqa python=3.8 -y
   conda activate conv_medvqa
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


# Model Download



# Data Construction 

Store the datasets in the /data/ref/ directory. After downloading the datasets, please organize the directory structure as follows.

```
└─ref
    ├─OVQA_publish
    ├─PathVQA
    ├─rad
    └─Slake1.0
```



# Training

```
# By adjusting the parameters in train.py, you can train various models across different datasets.
python train.py
```



# Testing

``` 
# By adjusting the test_best_model_path in test.py, you can load trained models for testing.
python test.py
```



