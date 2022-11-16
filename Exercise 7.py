import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data_square.csv').values
N = data.shape[0]
X = data[:,0].reshape(-1,1)
Y = data[:,1].reshape(-1,1)
plt.scatter(X,Y)
plt.xlabel('meter')
plt.ylabel('cost')
plt.grid()
plt.title('LINEAR REGRESSION')

#Building Xbar
X_2 = np.multiply(X,X)
Xbar = np.hstack((np.ones((N, 1)), X, X_2))

#Caculate weights of nonlinear regression
A = np.dot(Xbar.T, Xbar)
B = np.dot(Xbar.T, Y)
W = np.dot(np.linalg.pinv(A), B)
print(np.linalg.pinv(A))
print(W)


#Art

predict = np.dot(Xbar,W)
plt.plot((Xbar[:,1]), (predict), 'r')

#Result

plt.show()