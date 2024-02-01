# LDA

## About LDA

- Linear Discrimnant Analysis (aka Normal Discrimnant Analysis or Discrimnant Function Analysis) is a
dimensionality reduction technique used for supervised classification problems.

- It is used to project the features in higher dimension space into lower dimension space.

## Steps

1. Compute the class means of the dependent variable.
2. Derive the covariance matrix of the class variable.
3. Computer the within class - scatter matrix (S1+S2).
4. Compute the Eigen values and Eigen vectors.
5. Compute the Eigen values and Eigen vectors from the within
class and between class scatter matrix.

## Viva Questions

1) **What is LDA and PCA?**

   - LDA (Linear Discriminant Analysis): LDA is a dimensionality reduction and classification technique that maximizes the separation between classes by finding linear combinations of features.
   - PCA (Principal Component Analysis): PCA is a dimensionality reduction technique that transforms a dataset into a new coordinate system, capturing the most important information in the first few principal components.

2) **What is the most suitable application of LDA?**

   LDA is most suitable for classification problems where there are multiple classes, and the goal is to find a linear combination of features that maximizes the separation between classes.

3) **What is the difference between PCA and LDA?**

   - PCA is unsupervised and focuses on capturing overall variance in the data.
   - LDA is supervised and aims to maximize the separation between classes by finding the linear combination of features.

4) **How does LDA calculate its maximum separation?**

   LDA calculates the maximum separation by maximizing the ratio of the between-class variance to the within-class variance.

5) **Is LDA a supervised or unsupervised method?**

   LDA is a supervised method because it takes into account class labels during its training phase to find the linear combinations of features that best separate the classes.

6) **How do you estimate how much each variable contributes to the separation?**

   The contribution of each variable to the separation in LDA is estimated by analyzing the coefficients of the linear combination of features obtained during the LDA process.

7) **What metrics can be calculated with LDA?**

   Common metrics for LDA include accuracy, precision, recall, F1 score, and the confusion matrix, depending on the specific classification problem.

8) **What do the terms sensitivity and specificity mean?**

   - Sensitivity (True Positive Rate): The proportion of actual positive instances correctly identified by the classifier.
   - Specificity (True Negative Rate): The proportion of actual negative instances correctly identified by the classifier.

9) **How do you use Linear Discriminant Analysis as a classifier?**

   LDA can be used as a classifier by projecting new data points onto the linear discriminants obtained during training and assigning them to the class with the closest centroid.

10) **Can LDA be used as a multi-class classifier? If so, how would it work?**

    Yes, LDA can be used as a multi-class classifier. It works by considering each class as a separate entity and finding linear combinations of features that maximize the separation between all classes.

11) **Can Linear Discriminant Analysis be used for clustering?**

    While LDA is primarily a classification and dimensionality reduction technique, it can be adapted for clustering by treating the linear discriminants as features and applying clustering algorithms.

12) **What are the limitations of LDA?**

    - Assumes normality and equal covariance matrices for classes.
    - Sensitive to outliers.
    - May not perform well if classes are not linearly separable.
    - Requires the number of samples to be greater than the number of features.
