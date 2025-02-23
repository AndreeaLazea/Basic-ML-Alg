#-----------------------------SUPERVISED LEARNING-------------------------------
#                       = learning based on labeled data.
#For example, a labeled dataset of images of Elephant,
# Camel and Cow would have each image tagged with either “Elephant”
# , “Camel” or “Cow.”

#!!!!!!!Supervised learning is used in applications where historical data predicts likely future events.
#Unsupervised learning is used in applications where the goal is to understand the structure of the data.


#TYPES OF SUPERVISED LEARNING:
#
#1. --- Classification = learning to predict the category of a label
#                   EX: such as spam detection, image recognition, or fraud detection.
#
# TYPES OF CLASSIFICATION MODELS:
#   1.1 Logistic Regression
#   1.2 SVM (Support Vector Machine)
#   1.3 Decision Tree
#   1.4 Random Forest Classifier


#2. --- Regression = learning to predict a continuous value
#               EX: such as house prices, stock prices, or customer churn rates.
# TYPES OF REGRESSION MODELS:
#   2.1 Linear Regression
#   2.2 Polynomial Regression
#   2.3 Support Vector Machine Regression
#   2.4 Decision Tree Regression
#   2.5 Random Forest Regression

#EVALUATING SUPERVISED LEARNING MODELS:
#1. -------CLASSIFICATION METRICS:
# First we need to understand the Confusion Matrix.
# Confusion Matrix: 2x2 matrix that shows the number of:
# True Positives, False Positives, True Negatives, and False Negatives.
#   |------------------|
#   |   TP   |   FP    |
#   |------------------|
#   |   FN   |   TN    |
#   |------------------|
# TP = TRUE POSITIVE.
# TN = TRUE NEGATIVE
# FP = FALSE POSITIVE
# FN = FALSE NEGATIVE
#
#   1.1 Accuracy = % OF CORRECT PREDICTIONS, (TP+TN)/(TP+TN+FP+FN)
#   1.2 Precision = % OF POSITIVE PREDICTIONS THAT WERE CORRECT, TP/(TP+FP)
#   1.3 Recall = % OF ACTUAL POSITIVES THAT WERE CORRECTLY CLASSIFIED, TP/(TP+FN)
#   1.4 F1 Score = BALANCE BETWEEN PRECISION AND RECALL, 2*(Precision*Recall)/(Precision+Recall)
#   1.5 ROC-AUC = AREA UNDER THE RECEIVER OPERATING CHARACTERISTIC CURVE, TRUE POSITIVE RATE VS FALSE POSITIVE RATE
#
#Let's create a confusion matrix for a classification model:
#Implementation of Confusion Matrix for Binary classification using Python.
#PROBLEM: We have a list of actual labels and predicted labels.
#GOAL: Create a confusion matrix.
#SOLUTION:
#numpy = numerical python library for working with arrays.
#sklearn = machine learning library for python.
#seaborn = data visualization library for python.
#matplotlib = data visualization library for python.
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

actual = np.array(["Dog", "Dog", "Not Dog", "Dog", "Not Dog", "Not Dog", "Dog", "Not Dog", "Not Dog", "Not Dog"]);
predicted = np.array(["Dog", "Dog", "Dog", "Dog", "Not Dog", "Not Dog", "Dog", "Not Dog", "Dog", "Not Dog"]);

cm = metrics.confusion_matrix(actual, predicted, labels=["Dog", "Not Dog"]);
cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=["Dog", "Not Dog"]);
cm_display.plot();
plt.show();



#2. ---------- REGRESSION METRICS:
#   Regression metrics are used to evaluate the performance of regression models.
#Our aim is to get the calculated values as close
# to the actual values as possible
#   TO EVALUATE THAT WE CAN USE:

#   2.1 Mean Absolute Error (MAE):
# = measure of the average size of the errors in a set of predictions,
#   without taking their direction into account
#FORMULA: Sum from 1 to n of 1/n * |actual_value[i] - predicted_value[i]|
actual = [2, 3, 5, 5, 9]
predicted = [1, 3, 5, 7, 9]
n = 5
sum = 0;
for i in range(n):
    sum += abs(actual[i] - predicted[i])
mae = sum / n


#   2.2 Mean Squared Error (MSE)
# = measures how close a regression line is to a set of data points.
#   It is a risk function corresponding to the expected value of the
#   squared error loss.
#FORMULA: 1/n * sum from 1 to n of ( predicted_value[i] - actual_value[i])^2
sum = 0;
for i in range(n):
    sum += (predicted[i] - actual[i])**2
mse = sum/ n

#   2.3 Root Mean Squared Error (RMSE)
#sqare root of(1/n * sum from 1 to n of ( predicted_value[i] - actual_value[i])^2)
sum = 0
for i in range(n):
    sum += (predicted[i] - actual[i])**2
rmse = np.sqrt(sum/n)

#   2.4 R-Squared = coefficient of determination
# = measure of how well the regression line approximates
#    the real data points.
# FORMULA: R^2 = 1 - (sum of squares of residuals  / total sum of squares)
# * sum of squares of residuals =
#      sum from 1 to n of (actual_value[i] - predicted_value[i])^2
# * total sum of squares =
#       sum from 1 to n of (actual_value[i] - mean(actual_value))^2

sum_residuals = 0
sum_actual = 0
mean_actual = np.mean(actual)
for i in range(n):
    sum_residuals += (actual[i] - predicted[i])**2
    sum_actual += (actual[i] - mean_actual)**2
r_squared = 1 - (sum_residuals / sum_actual)

#   2.5 Adjusted R-Squared
# = R-Squared adjusted for the number of predictors in the model.
# FORMULA: R^2 = 1 - (1 - R^2) * (n-1) / (n-p-1)
#  * n = number of observations
#  * p = number of predictors
n = 5
p = 1
adjusted_r_squared = 1 - (1 - r_squared) * (n-1) / (n-p-1)



