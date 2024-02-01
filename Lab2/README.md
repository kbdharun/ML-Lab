# Principal Component Analysis

- Principal Component Analysis is a popular unsupervised learning technique for reducing the dimensionality of large
data sets.
- It increases interpretability yet, at the same time, it minimizes information loss. It helps to find the most significant
features in a dataset and makes the data easy for plotting in 2D and 3D. PCA helps in finding a sequence of linear combinations
of variables.

## Terms

- **Principal Component:** They are a straight line that captures most
of the variance of the data. They have a direction and
magnitude. Principal components are orthogonal projections
(perpendicular) of data onto lower-dimensional space.

- **Dimensionality:** Quantity of features or variables
used in the research.

## Steps

1. **Standardize the data**: PCA requires standardized data, so the first step is to standardize the
data to ensure that all variables have a mean of 0 and a standard deviation of 1.

2. **Calculate the covariance matrix**: The next step is to calculate the covariance matrix of the
standardized data. This matrix shows how each variable is related to every other variable in
the dataset.

3. **Calculate the eigenvectors and eigenvalues**: The eigenvectors and eigenvalues of the
covariance matrix are then calculated. The eigenvectors represent the directions in which
the data varies the most, while the eigenvalues represent the amount of variation along
each eigenvector.

4. **Choose the principal components**: The principal components are the eigenvectors with the
highest eigenvalues. These components represent the directions in which the data varies
the most and are used to transform the original data into a lower-dimensional space.

5. **Transform the data**: The final step is to transform the original data into the
lower-dimensional space defined by the principal components.

## Applications

- Used to visualize multidimensional data.
- Used to reduce the number of dimensions in healthcare data.
- Can help resize an image.
- Used in finance to analyze stock data and forecast returns.
- Helps to find patterns in the high-dimensional datasets.

## Viva Questions

1) **What do you mean by PCA?**

   Principal Component Analysis (PCA) is a dimensionality reduction technique used in statistics and machine learning
   to transform a dataset into a new coordinate system, where the variables are uncorrelated, and the most important 
   information is captured in the first few principal components.

2) **Uses of PCA**

   - Dimensionality reduction
   - Feature extraction
   - Noise reduction
   - Visualization of high-dimensional data

3) **How do you calculate the covariance matrix?**

   The covariance matrix is calculated by taking the covariance between each pair of variables in a dataset.

4) **How do you calculate Eigenvalues and Eigenvectors?**

   The eigenvalues and eigenvectors of a square matrix A can be found by solving the characteristic equation det(A - λI) = 0, where λ is the eigenvalue and I is the identity matrix. The corresponding eigenvector can be obtained by solving (A - λI)v = 0, where v is the eigenvector.

5) **Advantages & Disadvantages of PCA**

   - *Advantages:* Dimensionality reduction, noise reduction, visualization.
   - *Disadvantages:* Loss of interpretability, assumes linear relationships.

6) **Why are they called principal components?**

   The principal components are called so because they represent the directions in the data where the variance is maximized.

7) **Real-time applications of PCA?**

   - Face recognition
   - Image compression
   - Speech recognition
   - Financial market analysis

8) **Who is the counterpart of PCA in deep learning?**

   In deep learning, autoencoders are considered counterparts to PCA for dimensionality reduction.

9) **Why do we need dimensionality reduction? What are its drawbacks?**

   Dimensionality reduction is needed to overcome the curse of dimensionality and to improve model efficiency. Drawbacks include potential loss of information and interpretability.

10) **Is it important to standardize before applying PCA?**

    Yes, it is important to standardize the variables before PCA to give each variable equal weight in the analysis, as PCA is sensitive to the scale of the variables.

11) **Should one remove highly correlated variables before doing PCA?**

    Removing highly correlated variables before PCA is not necessary, as PCA can handle correlated variables. However, it may be done to simplify interpretation.

12) **What will happen when eigenvalues are roughly equal?**

    When eigenvalues are roughly equal, it indicates that the data has nearly equal variability along multiple dimensions, making it challenging to identify the most informative components.

13) **How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?**

    Evaluation can be done by comparing the performance of a machine learning model before and after dimensionality reduction using metrics like accuracy, precision, recall, or F1 score.

14) **Explain the Curse of Dimensionality?**

    The curse of dimensionality refers to the challenges and limitations that arise when dealing with high-dimensional data, including increased computational complexity, sparsity of data, and difficulty in visualization and interpretation.

15) **Can we implement Principal Component Analysis for Regression?**

    PCA can be implemented for regression by using the principal components as new features to build a regression model. However, it may result in reduced interpretability of the model.
