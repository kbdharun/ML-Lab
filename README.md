# ML-lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbdharun/ML-Lab)
[![CodeFactor](https://www.codefactor.io/repository/github/kbdharun/ml-lab/badge)](https://www.codefactor.io/repository/github/kbdharun/ml-lab)

This repository contains the programs that I worked out in Machine Learning Laboratory.

## Index

- Lab 1: Introduction to EDA
  - [Introduction to Matplotlib and Seaborn packages](https://github.com/kbdharun/ML-Lab/blob/main/Lab1/EDA_Matplotlib_&_Seaborn.ipynb)
  - [Introduction to Numpy and Pandas package](https://github.com/kbdharun/ML-Lab/blob/main/Lab1/Numpy_&_Pandas.ipynb)
  - [Insurance Data Analysis](https://github.com/kbdharun/ML-Lab/blob/main/Lab1/ML_Lab1_Insurance.ipynb)
  - [Iris Data Analysis](https://github.com/kbdharun/ML-Lab/blob/main/Lab1/ML_Lab1_Iris.ipynb)
- Lab 2: Principal Component Analysis
  - [About PCA](https://github.com/kbdharun/ML-Lab/blob/main/Lab2/README.md)
  - [PCA based dimensionality reduction on Wine Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab2/PCA-DR-Wine.ipynb)
  - [PCA using algorithm steps without `sklearn`](https://github.com/kbdharun/ML-Lab/blob/main/Lab2/PCA-using-alg-without-sk.ipynb)
  - [PCA using `sklearn` on Iris Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab2/PCA-using-sklearn-Iris.ipynb)
  - [PCA - Wine Quality Classification](https://github.com/kbdharun/ML-Lab/blob/main/Lab2/PCA-Wine-quality-classification.ipynb)
- Lab 3: K-Nearest Neighbors
  - [About KNN](https://github.com/kbdharun/ML-Lab/blob/main/Lab3/README.md)
  - [KNN using Iris Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab3/KNN-using-Iris.ipynb)
- Lab 4: Linear Discriminant Analysis and Linear Regression
  - [About LDA and LR](https://github.com/kbdharun/ML-Lab/blob/main/Lab4/README.md)
  - [LDA on Iris Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab4/LDA.ipynb)
  - [LR on Single Dataset (Iris)](https://github.com/kbdharun/ML-Lab/blob/main/Lab4/LR-on-single-dataset.ipynb)
  - [LR on Multiple Datasets (Iris, Wine)](https://github.com/kbdharun/ML-Lab/blob/main/Lab4/LR-on-multiple-datasets.ipynb)
- Lab 5: Logistic Regression
  - [About LR](https://github.com/kbdharun/ML-Lab/blob/main/Lab5/README.md)
  - [LR on Multiple Datasets (Iris, Wine)](https://github.com/kbdharun/ML-Lab/blob/main/Lab5/LR.ipynb)
- Lab 6: Naive Bayes Classifier
  - [About NBC](https://github.com/kbdharun/ML-Lab/blob/main/Lab6/README.md)
  - [NBC - Basic Program](https://github.com/kbdharun/ML-Lab/blob/main/Lab6/NBC.ipynb)
  - [NBC on Iris Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab6/NBC-Iris.ipynb)
  - [NBC Sentiment Analysis on IMDB Dataset (Short program)](https://github.com/kbdharun/ML-Lab/blob/main/Lab6/imdb-dataset-nbc-short-program.ipynb) (for exam)
  - [NBC Sentiment Analysis on IMDB Dataset (Detailed program)](https://github.com/kbdharun/ML-Lab/blob/main/Lab6/NBC-sentiment-analysis-IMDB.ipynb) (for understanding)

---

- Lab 7, 8: Support Vector Machine
  - [About SVM](https://github.com/kbdharun/ML-Lab/blob/main/Lab7/README.md)
  - [SVM Program on Breast Cancer Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab7/svm-breast-cancer.ipynb)
  - [SVM Program for Linear and non-Linear Classification Tasks on Breast Cancer Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab7/SVM.ipynb)
  - [SVM Program for Linear and non-Linear Classification Tasks on Iris Dataset](https://github.com/kbdharun/ML-Lab/blob/main/Lab8/SVM.ipynb)
- Lab 9: Multi-Layer Feed Forward Neural Network and Regularization Techniques
  - [About MLFFNN and Regularization Techniques](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/README.md)
  - [MLFFNN on Breast Cancer Dataset (Text)](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/FFNN-Text.ipynb) (not for exam)
  - [MLFFNN on MNIST Dataset (Image) - Short](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/FFNN-MNIST-Short.ipynb)
  - [MLFFNN on MNIST Dataset (Image) - Full](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/FFNN-MNIST-Full.ipynb)
  - [Regularization Techniques on Breast Cancer Dataset (Text)](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/Regularization.ipynb) (not for exam)
  - [Regularization Techniques on MNIST Dataset (Image)](https://github.com/kbdharun/ML-Lab/blob/main/Lab9/Regularization-MNIST.ipynb)

## Prerequisites

Python and packages in `requirements.txt` file installed.

> [!NOTE]
> You can install all the required packages using the command `pip install -r requirements.txt`.

### Working with Conda

If you are using `conda` to manage your environments, you can create a new environment for this repository with the command `conda create -n ml-lab` and activate it with the command `conda activate ml-lab`.

Then, you can install all the required packages using the command `conda install --file requirements.txt`.

### Container Image

Alternatively, you can use the [container image](https://github.com/kbdharun/ML-Lab/pkgs/container/ml-lab-image) I created with all the packages preinstalled.

You can install it in [Distrobox](https://github.com/89luca89/distrobox) with the command `distrobox create -i ghcr.io/kbdharun/ml-lab-image:latest -n ml` and use it with the command `distrobox enter ml`.

Additionally, you can verify the authenticity of the container image using [`cosign`](https://github.com/sigstore/cosign) (download the `cosign.pub` file from [here](https://github.com/kbdharun/ML-Lab/blob/main/cosign.pub) and execute the following command):

```zsh
cosign verify --key cosign.pub ghcr.io/kbdharun/ml-lab-image:latest
```
