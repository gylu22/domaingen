# dg


### Installation
Our code is based on [Detectron2](https://github.com/facebookresearch/detectron2) and requires python >= 3.6

Install the required packages
```
pip install -r requirements.txt
```

### Datasets
Set the environment variable DETECTRON2_DATASETS to the parent folder of the datasets

```
    path-to-parent-dir/
        /diverseWeather
            /daytime_clear
            /daytime_foggy 

```
Download [Diverse Weather](https://github.com/AmingWu/Single-DGOD) and [Cross-Domain](https://naoto0804.github.io/cross_domain_detection/) Datasets and place in the structure as shown.

### Training
We train our models on a single A100 GPU.
```
    python train.py --config-file configs/diverse_weather.yaml 


```

### Weights
[Download]() the trained weights.

### Citation


