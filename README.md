# ENEE633-Digit-Recognition
Project 2 for ENEE633 Statistical Pattern Recognition course at UMD

In this project, a Support Vector Machine (SVM) classifier and a Convolutional Neural Network (CNN) are implemented to classify the MNIST dataset.
- Dataset: MNIST dataset from http://yann.lecun.com/exdb/mnist/. It has a training set with 60000 28x28 gray-scale images of handwritten digits (10 classes). The testing set has 10000 images with the same size.
- SVM: The LIBSVM toolbox is used for the SVM classifier. Three different kernels (linear, polynomial, RBF) are evaluated. Before training, the data is reduced to a lower dimension using PCA and LDA methods.
- CNN: The Caffe toolbox is used for building the CNN.
