# Stealth-adversarial-attacks
This repository contains a sample code for generating stealth adversarial attacks on a custom network trained on MNIST dataset

The code builds on MATLAB deep learning tutorial and uses several functions supplied my MathWorks:

1) modelPredictions.m
2) preprocessMiniBatch.m
3) preprocessMiniBatchPredictors.m

The code assumes that the MNIST dataset (DigitDataset) is located in "\toolbox\nnet\nndemos\nndatasets\DigitDataset". Please make sure that the path is correct and if not either move the files or modify the path in "Example_Stealth_Attck.m" script

The main script is: Example_Stealth_Attack.m
