# cartoonGan
Generate cartoon avatar images using generative adversarial network 

# Usage
```
$ cd cartoonGan 
```
Training
```
$ mkdir pretrained_model
$ mkdir data/DATASET_NAME
  ... add images to data/DATASET_NAME/1 ...
$ python train.py --data_root data/DATASET_NAME
```
Testing
```
$ python test.py --model_root pretrained_model/ --model_name MODEL_NAME
```

# Dataset
Cartoon Set is a collection of random, 2D cartoon avatar images that can be downloaded from this link.
https://google.github.io/cartoonset/download.html

# Results
GAN Accuracy and Loss over 50 epochs

![image1](https://github.com/mohamedhaiham94/cartoonGan/blob/master/plots/plot.png)

Samples generted by the network

![image1](https://github.com/mohamedhaiham94/cartoonGan/blob/master/output/samples.gif)



