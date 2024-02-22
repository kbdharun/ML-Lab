# Naive Bayes Classifier

## About NBC

The Naive Bayes algorithm is a supervised machine learning algorithm based on the Bayes’ theorem.
It is a probabilistic classifier that is often used in NLP tasks like sentiment analysis (identifying a text
corpus’ emotional or sentimental tone or opinion).The Bayes’ theorem is used to determine the
probability of a hypothesis when prior knowledge is available.

## Advantages of NBC

- It is easy and fast to predict the class of the test data set. It also performs well in multi-class prediction.
- When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
- It performs well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).

## Disadvantages of NBC

- Naive Bayes is also known as a bad estimator, so the probability outputs are not to be taken too seriously.
- Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

## Algorithm NBC [Sentiment Analysis]

1. Loading the Dataset.
2. Data Preprocessing.
3. Encoding Labels and Making Train-Test Splits.
4. Building the Naive Bayes Classifier.
5. Fitting the Model on Training Set and Evaluating Accuracies on the Test Set.

## Viva Questions

1. **What mathematical concept Naive Bayes is based on?**
   **Ans.** Naive Bayes is based on the mathematical concept of Bayes' Theorem.

2. **What are the different types of Naive Bayes classifiers?**
   **Ans.** Different types of Naive Bayes classifiers include Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes.

3. **Is Naive Bayes a classification algorithm or regression algorithm?**
   **Ans.** Naive Bayes is a classification algorithm.

4. **What are some benefits of Naive Bayes?**
   **Ans.** Benefits of Naive Bayes include simplicity, efficiency, and effectiveness in high-dimensional spaces.

5. **What are the cons of Naive Bayes classifier?**
   **Ans.** Cons of Naive Bayes include the assumption of independence between features, which may not hold true in all cases.

6. **What are the applications of Naive Bayes?**
   **Ans.** Naive Bayes is used in applications like spam filtering, sentiment analysis, and document categorization.

7. **Is Naive Bayes a discriminative classifier or generative classifier?**
   **Ans.** Naive Bayes is a generative classifier.

8. **What is the formula given by Bayes theorem?**
   **Ans.** Bayes' Theorem is given by \( P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \).

9. **What is posterior probability and prior probability in Naïve Bayes?**
   **Ans.** In Naïve Bayes, posterior probability is the probability of the class given the features, and prior probability is the probability of the class without considering the features.

10. **What’s the difference between Generative Classifiers and Discriminative Classifiers?**
    **Ans.** Generative classifiers model the joint probability distribution of features and labels, while discriminative classifiers model the conditional probability of labels given features.
