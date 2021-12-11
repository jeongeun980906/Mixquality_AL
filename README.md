# Mixquality_AL

Final Term Project For Bayesian Machine Learning Lecture (XAI-623)

## Youtube Link

The presentation is given in 
[YoutubeLink](https://www.youtube.com/watch?v=yUbdp-9-lj4)

## Problem Formulation
**Dirty MNIST**

Active learning on MNIST: Ambiguous MNIST = 1:60

which is same with [DDU](https://arxiv.org/pdf/2102.11582.pdf) paper. 

**OOD MNIST**

Active learning on MNIST: Ambiguous MNIST : EMNIST = 1:60:1
<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Mixquality_AL/blob/main/misc/problem.png">
</p>

## Method
use MLN + Feature Density
Traning process of MLN is same as [paper](https://arxiv.org/abs/2111.01632) and [implementation](https://github.com/jeongeun980906/Uncertainty-Aware-Robust-Learning).
Estimation of feature density is done by modeling GMM with EM algorithm, implemented [here](https://github.com/SeungyounShin/DDU_pytorch)

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Mixquality_AL/blob/main/misc/method.png">
</p>

## Expriment Results
Dirty MNIST
<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Mixquality_AL/blob/main/misc/dirty_mnist_2.png">
</p>

OOD MNIST
<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Mixquality_AL/blob/main/misc/ood_mnist_2.png">
</p>

