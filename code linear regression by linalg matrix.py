
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180,
               183]]).reshape(-1,1)
Y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

#Show result
plt.scatter(X, Y)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear regression")


#Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.hstack((one,X))


#Caculate weights of the linear regression model
A = np.dot(Xbar.T, Xbar)
B = np.dot(Xbar.T, Y)
W = np.dot(np.linalg.pinv(A), B)
print(W)

#Art line
predict = np.dot(Xbar, W.reshape(-1,1))
plt.plot((Xbar[0][1], Xbar[X.shape[0]-1][1]), (predict[0], predict[X.shape[0]-1]), 'r')

#weights
w_0, w_1 = W[0], W[1]
y1 = w_1*155 + w_0
y2 = w_1*160 + w_0
print('Input 155cm, true output 52kg, predicted output %.2fkg.' %(y1))
print('Input 165cm, true output 56kg, predicted output %.2fkg.' %(y2))

plt.show()


