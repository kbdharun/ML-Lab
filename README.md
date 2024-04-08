# ML-lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbdharun/ML-Lab)
[![CodeFactor](https://www.codefactor.io/repository/github/kbdharun/ml-lab/badge)](https://www.codefactor.io/repository/github/kbdharun/ml-lab)

This repository contains the programs that I worked out in Machine Learning Laboratory.

## Index

- Lab 1: Introduction to EDA
  - [Introduction to Matplotlib and Seaborn packages](Lab01/EDA_Matplotlib_&_Seaborn.ipynb)
  - [Introduction to Numpy and Pandas package](Lab01/Numpy_&_Pandas.ipynb)
  - [Insurance Data Analysis](Lab01/ML_Lab1_Insurance.ipynb)
  - [Iris Data Analysis](Lab01/ML_Lab1_Iris.ipynb)
- Lab 2: Principal Component Analysis
  - [About PCA](Lab02/README.md)
  - [PCA based dimensionality reduction on Wine Dataset](Lab02/PCA-DR-Wine.ipynb)
  - [PCA using algorithm steps without `sklearn`](Lab02/PCA-using-alg-without-sk.ipynb)
  - [PCA using `sklearn` on Iris Dataset](Lab02/PCA-using-sklearn-Iris.ipynb)
  - [PCA - Wine Quality Classification](Lab02/PCA-Wine-quality-classification.ipynb)
- Lab 3: K-Nearest Neighbors
  - [About KNN](Lab03/README.md)
  - [KNN using Iris Dataset](Lab03/KNN-using-Iris.ipynb)
- Lab 4: Linear Discriminant Analysis and Linear Regression
  - [About LDA and LR](Lab04/README.md)
  - [LDA on Iris Dataset](Lab04/LDA.ipynb)
  - [LR on Single Dataset (Iris)](Lab04/LR-on-single-dataset.ipynb)
  - [LR on Multiple Datasets (Iris, Wine)](Lab04/LR-on-multiple-datasets.ipynb)
- Lab 5: Logistic Regression
  - [About LR](Lab05/README.md)
  - [LR on Multiple Datasets (Iris, Wine)](Lab05/LR.ipynb)
- Lab 6: Naive Bayes Classifier
  - [About NBC](Lab06/README.md)
  - [NBC - Basic Program](Lab06/NBC.ipynb)
  - [NBC on Iris Dataset](Lab06/NBC-Iris.ipynb)
  - [NBC Sentiment Analysis on IMDB Dataset (Short program)](Lab06/NBC-IMDB-short-program.ipynb) (for exam)
  - [NBC Sentiment Analysis on IMDB Dataset (Detailed program)](Lab06/NBC-sentiment-analysis-IMDB.ipynb) (for understanding)

---

- Lab 7, 8: Support Vector Machine
  - [About SVM](Lab07/README.md)
  - [SVM Program on Breast Cancer Dataset](Lab07/svm-breast-cancer.ipynb)
  - [SVM Program for Linear and non-Linear Classification Tasks on Breast Cancer Dataset](Lab07/SVM.ipynb)
  - [SVM Program for Linear and non-Linear Classification Tasks on Iris Dataset](Lab08/SVM.ipynb)
- Lab 9: Multi-Layer Feed Forward Neural Network and Regularization Techniques
  - [About MLFFNN and Regularization Techniques](Lab09/README.md)
  - [MLFFNN on Breast Cancer Dataset (Text)](Lab09/FFNN-Text.ipynb) (not for exam)
  - [MLFFNN on MNIST Dataset (Image) - Short](Lab09/FFNN-MNIST-Short.ipynb)
  - [MLFFNN on MNIST Dataset (Image) - Full](Lab09/FFNN-MNIST-Full.ipynb)
  - [Regularization Techniques on Breast Cancer Dataset (Text)](Lab09/Regularization-Text.ipynb) (not for exam)
  - [Regularization Techniques on MNIST Dataset (Image)](Lab09/Regularization-MNIST.ipynb)
  - [Regularization Techniques with comparision on Diabetes Dataset (Text)](Lab09/Regularization-comp.ipynb) ([Alternative program](Lab09/Regularization1.ipynb))
  - [Regularization Techniques on Obesity Classification Dataset (Text)](Lab09/Regularization-Obesity.ipynb)
- Lab 10, 11: Artificial Neural Network, Convolutional Neural Network; Hidden Markov Model based techinques (Viterbi Algorithm, Trellis, Long Short Term Memory)
  - [About ANN, CNN & HMM](Lab10,11/README.md)
  - [ANN on CIFAR10 Dataset](Lab10,11/Img-Classification-ANN-CIFAR10.ipynb)
  - [CNN on CIFAR10 Dataset](Lab10,11/Img-Classification-CNN-CIFAR10.ipynb) ([Alternative program](Lab10,11/NN_Image_Classifications.ipynb))
  - [HMM - Viterbi Algorithm on own Weather Dataset](Lab10,11/HMM.ipynb) ([Alternative program](Lab10,11/HMM-sample.ipynb))
  - [HMM - Viterbi Algorithm (with Trellis) on Iris Dataset](Lab10,11/HMM-Viterbi,Trellis.ipynb)
  - [HMM - LSTM Algorithm on Iris Dataset](Lab10,11/HMM-LSTM.ipynb) (not for exam)

## Prerequisites

Python and packages in `requirements.txt` file installed.

> [!NOTE]
> You can install all the required packages using the command `pip install -r requirements.txt`.

### Working with Conda

If you are using `conda` to manage your environments, you can create a new environment for this repository with the command `conda create -n ml-lab` and activate it with the command `conda activate ml-lab`.

> [!TIP]
> For faster environment solving in Conda, I would suggesting using the `libmamba` solver. You can set it as the default solver using the command `conda config --set solver libmamba`.

Then, you can install all the required packages using the command `conda install --file requirements.txt`.

### Container Image

Alternatively, you can use the [container image](https://github.com/kbdharun/ML-Lab/pkgs/container/ml-lab-image) I created with all the packages preinstalled.

You can install it in [Distrobox](https://github.com/89luca89/distrobox) with the command `distrobox create -i ghcr.io/kbdharun/ml-lab-image:latest -n ml` and use it with the command `distrobox enter ml`.

Additionally, you can verify the authenticity of the container image using [`cosign`](https://github.com/sigstore/cosign) (download the `cosign.pub` file from [here](cosign.pub) and execute the following command):

```zsh
cosign verify --key cosign.pub ghcr.io/kbdharun/ml-lab-image:latest
```
