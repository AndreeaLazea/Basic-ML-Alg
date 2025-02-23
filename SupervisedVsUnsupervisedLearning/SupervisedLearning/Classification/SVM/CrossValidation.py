# Cross Validation
# = technique used in ML to evaluate the performance of a model on unseen data.
# ----- HOW IT WORKS?
#  - The data is decided into multiple folds or subsets
#    using one of these folds as a validation set,
#    and the rest as training set.

# ----- WHAT IS IT USED FOR?
#  - Avoid overfitting by training the model on different subsets of the data.

# ----- TYPES OF CROSS VALIDATION
# 1. K-Fold Cross Validation
# = The dataset is divided into K subsets or folds.
# - The model is trained on K-1 folds and tested on the remaining fold.
# - 10 should be the lowest value for K as lower or higher values can lead to LOOCV


# 2. LOOCV - Leave One Out Cross Validation
# = The model is trained on all data points except one, which is used for testing.
# - Advantage: We make use of all data points => low bias.
# - Drawback: Major one is that it leads to higher variation in the testing model
#             performance because we test against one data point.
#             If the data point is an outlier, the model will be biased.


# 3. Holdout Validation
# = Perform training on the 50% of the given dataset and the rest is for testing.
# - Drawbacks: Possible that the remaining 50% of the data contains some
#              important information that is not present in the 50% of the data
#              used for training.


# 4. Stratified Cross Validation
# = Ensures that each fold is representative of all strata of the data.
# - Strata: Subsets of the data that are similar to each other but different
#            from other subsets.
# - Example: If a dataset has 25% of data points of class A and 75% of class B,
#            stratified cross-validation ensures that each fold has 25% of class A
#            and 75% of class B.
# Advantage: Particularly useful when dealing with imbalanced datasets.


#EXAMPLE:
#STEPS:
#1. Import the required libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC #Support Vector Classifier
from sklearn.datasets import load_iris

#2. Load the dataset
data = load_iris()

#3. Preprocess the data
X = data.data
y = data.target

#4. Create SVM classifier
svm_classifier = SVC(kernel='linear')

#5. Define the number of folds for class-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=23)

#6. Perform k-folds cross-validation
cross_Val_Results = cross_val_score(svm_classifier, X, y, cv = kf)
#cv = kf is the number of folds

#7. Evaluation metrics
print("Cross Validation Results: ", cross_Val_Results)
print("Mean Accuracy: ", cross_Val_Results.mean())


