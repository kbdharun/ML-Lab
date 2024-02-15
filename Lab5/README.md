# Logistic Regression

- Logistic regression is one of the most popular Machine Learning algorithms,
which comes under the Supervised Learning technique. It is used for predicting
the categorical dependent variable using a given set of independent variables.
- Logistic regression predicts the output of a categorical dependent variable.
Therefore the outcome must be a categorical or discrete value. It can be either
Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0
and 1, it gives the probabilistic values which lie between 0 and 1.
- Logistic Regression is much similar to the Linear Regression except that how
they are used. Linear Regression is used for solving Regression problems,
whereas Logistic regression is used for solving the classification problems.
- In Logistic regression, instead of fitting a regression line, we fit an "S" shaped
logistic function, which predicts two maximum values (0 or 1).

## Steps

### Overview

- Data Pre-processing step
- Fitting Logistic Regression to the Training set
- Predicting the test result
- Test accuracy of the result(Creation of Confusion matrix)
- Visualizing the test set result.

### Detailed

- **Step 1**: Import Packages
- **Step 2a**: Get Data
- **Step 2b**: Split Data
- **Step 2c**: Scale Data
- **Step 3**: Create a Model and Train It
- **Step 4**: Evaluate the Model

## Viva Questions

1. **Is logistic regression a generative or a descriptive classifier? Why?**
   **Ans.** Logistic regression is a discriminative classifier. It models the decision boundary directly rather than modeling the distribution of each class.

2. **Can you use logistic regression for classification between more than two classes?**
   **Ans.** Yes, logistic regression can be extended for classification between more than two classes using techniques like one-vs-all (OvA) or one-vs-one (OvO) strategies.

3. **How do you implement multinomial logistic regression?**
   **Ans.** Multinomial logistic regression is implemented by using the `multi_class` parameter in scikit-learn's `LogisticRegression`, typically set to 'multinomial'. It extends binary logistic regression to handle multiple classes.

4. **Why can't we use the mean square error cost function used in linear regression for logistic regression?**
   **Ans.** Mean square error cost function is not used in logistic regression because the resulting optimization problem is non-convex and may have multiple local minima, making it challenging to find the global minimum.

5. **What alternative could you suggest using a for loop when using Gradient Descent to find the optimum parameters for logistic regression?**
   **Ans.** An alternative to using a for loop in Gradient Descent optimization for logistic regression is vectorization, leveraging NumPy operations for efficient computations on entire arrays simultaneously.

6. **What are the different types of Logistic Regression?**
   **Ans.** Types of logistic regression include binary logistic regression, multinomial logistic regression, and ordinal logistic regression.

7. **Is the decision boundary Linear or Non-linear in the case of a Logistic Regression model?**
   **Ans.** The decision boundary in logistic regression is linear, regardless of the dimensionality of the input features.

8. **What is the Impact of Outliers on Logistic Regression?**
   **Ans.** Outliers can have a significant impact on logistic regression by influencing the estimated coefficients and affecting the position of the decision boundary, potentially leading to suboptimal model performance.

9. **What is the difference between the outputs of the Logistic model and the Logistic function?**
   **Ans.** The output of the logistic model refers to the predicted log-odds, while the logistic function transforms these log-odds into probabilities.

10. **How do we handle categorical variables in Logistic Regression?**
    **Ans.** Categorical variables in logistic regression are handled by encoding them into numerical values using techniques like one-hot encoding or label encoding.

11. **What are the assumptions made in Logistic Regression?**
    **Ans.** Assumptions in logistic regression include linearity, independence of errors, absence of multicollinearity, and a large sample size for stable parameter estimates.

12. **Why is Logistic Regression termed as regression and not classification?**
    **Ans.** Logistic regression is termed as regression due to its historical context, where it originated as a method for regression analysis. Despite its name, it is widely used for binary and multinomial classification problems.
