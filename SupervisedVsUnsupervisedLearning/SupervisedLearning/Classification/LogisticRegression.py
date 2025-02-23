#1. LOGISTIC REGRESSION
# a supervised ML algorithm used for classification
# the goal is to predict the probability that an
#instance belongs to  agiven class or not.
# it's a statistial algorithm, which analyzes the relationship
#between two data factors.
# WHEN IS IT USED?
# - binary classification problems, where we use SIGMOID FUNCTION
#   that takes input as independent variables and produces a probability
#   value between 0 and 1. If the probability is greater than 0.5, the
#   instance is classified as 1, otherwise as 0.
from random import random

# If it's a classification problem, why is it refered to as regression?
# = bc it's the extension of linear regression, used in classification problems

#TYPES OF LOGISTIC REGRESSION
# - Binomial Logistic Regression = 2 classes, 0 or 1, Pass or Fail etc.
# - Multinomial Logistic Regression = 3 or more classes, 0, 1, 2, or Cat, Dog, Elephant etc.
# - Ordinal Logistic Regression = 3 or more classes with an order, 0, 1, 2 or Low, Medium, High etc.

#ASSUMPTIONS OF LOGISTIC REGRESSION
# Understanding these assumptions is important to ensure that we are using
#         appropriate application of the model.
#The assumptions are:
#  1. Independent Observations:
#     - The observations are independent of each other. There is no correlation between any input variables
#  2. Binary Dependent Variables:
#     - the dependent variable must be binary, meaning it has only two possible outcomes.
#  3. Linearity Relationship between Independent Variables and Log Odds:
#     - The independent variables should have a linear relationship with the log odds.
#  4. No Outliers:
#     - There should be no outliers in the data.
#  5. Large Sample Size:
#     - The sample size should be large enough to ensure the model's stability.

# ---------UNDERSTANDING SIGMOID FUNCTION
# Very important to understand, as it's the core of logistic regression.
# = Mathematical function used to map the predicted values to probabilties.
# = It maps any real value into a value between 0 and 1.
# The value of the logistic regression must be between 0 and 1,
#which cannot go beyond this limit so it forms a curve like the S form.
# FORMULA: f(x) = 1 / (1 + e^-x)
# where x = independent variable


# ---------UNDERSTANDING ODDS RATIO
# = measure of the relationship between the target and independent variables.
# = odds of an event happening compared to the odds of the event not happening.
# FORMULA: e^z = p / (1-p)
# where p = probability of the event happening
# If the odds ratio is greater than 1, the event is more likely to occur.
# If the odds ratio is less than 1, the event is less likely to occur.

# ---------TERMINOLOGY INVOLVING LOGISTIC REGRESSION
#1. Independent variables: input characteristics applied to the dependent variable
#2. Dependent variable: the outcome we want to predict
#3. Logistic function:  formula used to present how the independent and dependent
#                       variables are related.
#                      ex: f(x) = 1 / (1 + e^-x)
#4. Odds : ratio of the probability of an event happening to the probability of the event not happening
#5. Log-Odds: the natural logarithm of the odds
#6. Coefficient: the logistic regression mode;'s estimated parameters,
#                show how the independent and dependent variables relate
#                to each other.
#7. Intercept:  Constant term in the logistic regression model,
#               represents the estimated log-odds of the dependent variable
#               when all independent variables are zero.
#8. Maximum likelihood estimate: method used to estimate the coefficients of the logistic regression model,
#                                by maximizing the likelihood of observing the data given in the model.

#----------IMPLEMENTATION OF LOGISTIC REGRESSION
# STEPS:
#1. Import the required libraries
#2. Load the dataset
#3. Preprocess the data
#4. Split the dataset into training and testing sets
#5. Create a logistic regression model
#6. Train the model
#7. Make predictions
#8. Evaluate the model
#9. Improve the model
#10. Make predictions

#IMPLEMENTATION OF EXAMPLE 1:
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. Loading the data
data = load_breast_cancer()

#2. Preprocess the data
X = data.data
y = data.target

#3. Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=23)
#FUN FACT: random state is important bc it ensures that the data is split
#in the same way each time the code is run

#4. Creating the model
logistic_model = LogisticRegression(max_iter=10000, random_state=0)

#5. Training the model
logistic_model.fit(X_train, y_train)

#6. Making predictions
y_pred = logistic_model.predict(X_test)

#7. Evaluating the model
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy: ", accuracy)


#EXAMPLE 2: FOR MULTINOMIAL LOGISTIC REGRESSION:
#Multinomial Logistic Regression is used when the
# dependent variable has more than two categories.

#IMPLEMENTATION:
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

#1. Loading the data
digits = datasets.load_digits()

#2. Preprocess the data
X = digits.data
y = digits.target

#3. Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1);

#4. Create the model
reg_model = LogisticRegression(max_iter=10000, random_state=0)

#5. Train the model

reg_model.fit(X_train, y_train)

#6. Make predictions

y_pred = reg_model.predict(X_test)

#7. Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("Accuracy: ", accuracy)


