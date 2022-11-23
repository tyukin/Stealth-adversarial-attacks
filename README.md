# Stealth-adversarial-attacks
This repository contains a sample code for generating stealth adversarial attacks on a custom network trained on MNIST dataset

The code builds on MATLAB deep learning tutorial and uses several functions supplied my MathWorks:

1) modelPredictions.m
2) preprocessMiniBatch.m
3) preprocessMiniBatchPredictors.m

The code assumes that the MNIST dataset (DigitDataset) is located in "\toolbox\nnet\nndemos\nndatasets\DigitDataset". The \DigitDataset folder containes 10 subfulders with images for digits 0 - 9. Each subfulder contains 1000 images of a digit. Please make sure that the path is correct. If the dataset is in a different directory then please either move the files to "\toolbox\nnet\nndemos\nndatasets\DigitDataset" or modify the path in "Example_Stealth_Attack_script.m" script. 

The main script is: Example_Stealth_Attack_script.m


