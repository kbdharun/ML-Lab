# Principal Component Analysis

- Principal Component Analysis is a
popular unsupervised learning technique
for reducing the dimensionality of large
data sets.
- It increases interpretability yet,
at the same time, it minimizes information
loss. It helps to find the most significant
features in a dataset and makes the data
easy for plotting in 2D and 3D. PCA helps
in finding a sequence of linear combinations
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
