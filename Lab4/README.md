# LDA and LR

## Index

- [LDA](#lda)
  - [About LDA](#about-lda)
  - [Steps](#steps)
  - [Viva Questions](#viva-questions)
- [LR](#lr)
  - [About LR](#about-lr)
  - [Steps](#steps-1)
  - [Viva Questions](#viva-questions-1)

## LDA

### About LDA

- Linear Discrimnant Analysis (aka Normal Discrimnant Analysis or Discrimnant Function Analysis) is a
dimensionality reduction technique used for supervised classification problems.

- It is used to project the features in higher dimension space into lower dimension space.

### Steps

1. Compute the class means of the dependent variable.
2. Derive the covariance matrix of the class variable.
3. Computer the within class - scatter matrix (S1+S2).
4. Compute the Eigen values and Eigen vectors.
5. Compute the Eigen values and Eigen vectors from the within
class and between class scatter matrix.

### Viva Questions

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

## LR

### About LR

- Linear Regression is a supervised machine learning algorithm that is used to predict the value of a continuous variable based on one or more predictor variables.

- It is used to predict the value of a continuous variable based on one or more predictor variables.

### Steps-1

- Problem Understanding: Identify variables and assess linearity.
- Data Collection and Preparation: Gather, handle missing values, and split into sets.
- Explore Data: Conduct exploratory data analysis.
- Model Selection: Choose simple or multiple linear regression.
- Data Preparation for Modeling: Extract Y and X, standardize if necessary.
- Data Splitting: Divide into training/testing sets.
- Model Building: Fit linear regression using training data.
- Model Evaluation: Assess performance with testing set metrics.
- Results Interpretation: Understand coefficients and variable impact.
- Model Fine-tuning: Adjust based on evaluation insights.
- Predictions: Utilize the model for new data predictions.

### Viva Questions-1

1) **What is the output of Linear Regression in machine learning?**

   The output of Linear Regression is a linear equation that represents the relationship between the independent variable(s) and the dependent variable.

2) **What are the benefits of using Linear Regression?**

   - Simplicity and interpretability
   - Efficient for linear relationships
   - Provides insights into variable importance

3) **What are the assumptions of a linear regression model?**

   - Linearity
   - Independence of residuals
   - Homoscedasticity (constant variance of residuals)
   - Normality of residuals

4) **What are outliers? How do you detect and treat them? How do you deal with outliers in a linear regression model?**

   - Outliers are extreme values that deviate from the overall pattern of the data.
   - Detection methods: visual inspection, statistical tests.
   - Treatment: removing, transforming, or assigning lower weight.

5) **How do you determine the best fit line for a linear regression model?**

   The best fit line is determined by minimizing the sum of squared differences between the observed and predicted values, usually using the method of least squares.

6) **What is the difference between simple and multiple linear regression?**

   - Simple linear regression involves one independent variable.
   - Multiple linear regression involves two or more independent variables.

7) **What is linear Regression Analysis?**

   Linear Regression Analysis is a statistical method to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

8) **What is multicollinearity and how does it affect linear regression analysis?**

   Multicollinearity is the presence of high correlation between independent variables. It can lead to unstable coefficient estimates and reduced interpretability in linear regression analysis.

9) **What is the difference between linear regression and logistic regression?**

   - Linear regression predicts continuous outcomes.
   - Logistic regression predicts the probability of a binary outcome.

10) **What are the common types of errors in linear regression analysis?**

    - Residuals or errors: the differences between observed and predicted values.
    - Common types include mean squared error (MSE) and root mean squared error (RMSE).

11) **How do you measure the strength of a linear relationship between two variables?**

    The strength of a linear relationship is measured by the correlation coefficient, with values close to +1 or -1 indicating a strong relationship.

12) **What is the difference between linear regression and non-linear regression?**

    - Linear regression models linear relationships.
    - Non-linear regression models non-linear relationships.

13) **What are the common techniques used to improve the accuracy of a linear regression model?**

    - Feature engineering
    - Handling outliers
    - Polynomial regression
    - Regularization techniques (e.g., Ridge, Lasso)

14) **How do you evaluate the goodness of fit of a linear regression model?**

    - R-squared (coefficient of determination)
    - Residual analysis
    - F-statistic

15) **How to find RMSE and MSE?**

    - MSE (Mean Squared Error) is calculated as the average of the squared differences between observed and predicted values.
    - RMSE (Root Mean Squared Error) is the square root of MSE and provides the average error in the same units as the dependent variable.
