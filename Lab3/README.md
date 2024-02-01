# KNN

## About KNN

- The k-nearest neighbors algorithm, also known as KNN or k-NN, is a
non-parametric, supervised learning classifier, which uses proximity to make
classifications or predictions about the grouping of an individual data point.
- While it can be used for either regression or classification problems, it is
typically used as a classification algorithm,
- The K-NN algorithm compares a new data entry to the values in a given
data set (with different classes or categories).
- Based on its closeness or similarities in a given range (K) of neighbors, the
algorithm assigns the new data to a class or category in the data set
(training data).

## Steps

1. Assign a value to K.

2. Calculate the distance between the new data entry and all
other existing data entries (Use Euclidean distance). Arrange them in
ascending order.

3. Find the K nearest neighbors to the new entry based on the
calculated distances.

4. Assign the new data entry to the majority class in the nearest
neighbors.

## Viva Questions

1) **Why is KNN a non-parametric Algorithm?**

   KNN is non-parametric because it does not make any assumptions about the underlying data distribution. It does not have a fixed set of parameters; instead, it learns from the data during the training phase.

2) **What is “K” in the KNN Algorithm?**

   “K” in the KNN Algorithm represents the number of nearest neighbors considered when making a prediction for a new data point.

3) **Why is the odd value of “K” preferred over even values in the KNN Algorithm?**

   An odd value of “K” is preferred to avoid ties when voting for the class of the new data point. With an odd “K,” there will always be a clear majority.

4) **How does the KNN algorithm make predictions on the unseen dataset?**

   KNN predicts the class of a new data point by considering the majority class among its K nearest neighbors based on a distance metric (e.g., Euclidean distance).

5) **Is Feature Scaling required for the KNN Algorithm? Explain with proper justification.**

   Yes, feature scaling is required for KNN as it is sensitive to the scale of the features. If features have different scales, those with larger scales may dominate the distance calculations.

6) **What is the space and time complexity of the KNN Algorithm?**

   - Space Complexity: O(n) for storing the training dataset.
   - Time Complexity: O(d * n * log(n)) for sorting distances, where n is the number of training samples and d is the number of features.

7) **Can the KNN algorithm be used for regression problem statements?**

   Yes, KNN can be used for regression by predicting the average (or weighted average) of the target values of the K nearest neighbors.

8) **Why is the KNN Algorithm known as Lazy Learner?**

   KNN is called a Lazy Learner because it doesn't learn a model during the training phase. It memorizes the entire training dataset and performs the actual computation during prediction.

9) **Why is it recommended not to use the KNN Algorithm for large datasets?**

   KNN's prediction time increases with the size of the training dataset since it requires computing distances between the new data point and all training points.

10) **How to handle categorical variables in the KNN Algorithm?**

    Categorical variables in KNN can be handled by converting them into numerical values or using distance metrics suitable for categorical data.

11) **How to choose the optimal value of K in the KNN Algorithm?**

    The optimal value of K can be chosen through techniques like cross-validation, where different values of K are tested, and the one with the best performance is selected.

12) **How can you relate KNN Algorithm to the Bias-Variance tradeoff?**

    In KNN, a small value of K may lead to high variance and low bias, resulting in overfitting, while a large K may lead to high bias and low variance, resulting in underfitting.

13) **Which algorithm can be used for value imputation in both categorical and continuous categories of data?**

    KNN can be used for value imputation in both categorical and continuous data by predicting missing values based on the K nearest neighbors.

14) **Is it possible to use the KNN algorithm for Image processing?**

    Yes, KNN can be used for image processing tasks such as image recognition by considering pixel values as features and finding similar images.

15) **What are the real-life applications of KNN Algorithms?**

    - Recommender systems
    - Handwriting recognition
    - Fraud detection
    - Medical diagnosis
    - Text categorization
    - Image recognition
