
import numpy as np
from sklearn import datasets, linear_model


X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180,
               183]]).reshape(-1,1)
Y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# fit the model by Linear regression
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# Compare two results
print("scikit-learn's solution: w_1 = ", regr.coef_[0], "w_0 = ",
      regr.intercept_)