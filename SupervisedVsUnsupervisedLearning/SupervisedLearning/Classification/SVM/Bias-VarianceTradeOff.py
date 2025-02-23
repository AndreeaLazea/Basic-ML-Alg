#What is Bias?
#  = Difference between the prediction of the values by the ML model and the correct value.
# - High bias gives large errors in training as well as testing data.
# - Recommended to have a low Bias for avoiding underfitting.
#       * High bias => straight line, not fitting the model well.
#                      Happens when hypothesis is too simple or linear in nature
#

#What is Variance?
#  = The variability of the model prediction for a given data point.
# - High variance gives large errors in testing data.
# - Recommended to have a low variance for avoiding overfitting.
#       * High variance => wavy line, fitting the model too well.
#                         Happens when hypothesis is too complex or non-linear in nature.
#

#What is Bias-Variance Tradeoff?
# IF alg too simple => HIGH BIAS, LOW VARIANCE => ERROR prone :(
# IF alg too complex => LOW BIAS, HIGH VARIANCE => ERROR prone :(
#IF LOW BIAS, LOW VARIANCE => GOOD MODEL :)
# The goal is to find the optimal balance between bias and variance.