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
