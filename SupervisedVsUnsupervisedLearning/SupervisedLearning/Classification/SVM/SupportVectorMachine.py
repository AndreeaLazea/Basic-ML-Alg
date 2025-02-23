#2. SVM - SUPPORT VECTOR MACHINE
# = A supervised machine learning algorithm used for classification and regression,
# but mostly used for classification.
# = Find optimal hyperplane in an N-dimensional space to separate
# data points into different classes.
# = Maximizes the margin between the closest points of different classes.
# = The points closest to the hyperplane are called support vectors.

# ----------- SVM TERMINOLOGY
#1. Hyperplane: decision boundary separating different classes in feature space
#               represented by the equation w*x + b = 0 in linear classification.
#2. Support Vectors: data points closest to the hyperplane, crucial for determening
#                    the hyperplane and margin in SVM.
#3. Margin: distance between the hyperplane and the support vectors.
#4. Kernel: function that maps data to a higher-dimensional space,
#           making it easier to classify non-linear separable data.
#5. Hard Margin: A maximum-margin hyperplane that perfectly separates the data without
#                any misclassification.
#6. Soft Margin: Allows missclassifications by introducing slack variables
#                balancing margin maximization and misclassification penalties
#                when data is not perfectly linearly separable.
#7. C: Regularization term balancing margin maximization and misclassification penalties.
#      Higher C value enforces a stricter penalty for classification.