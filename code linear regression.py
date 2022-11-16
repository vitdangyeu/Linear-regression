import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)
plt.scatter(x,y)
plt.xlabel('meter')
plt.ylabel('cost')
plt.grid()
plt.title('LINEAR REGRESSION')

x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 1.]).reshape(-1,1)

numOfiteration = 100
learning_rate = 0.000001
cost = np.zeros((numOfiteration, 1))

for i in range(numOfiteration):
    r = np.dot(x,w) - y
    cost[i] = 0.5*np.sum(r*r)/N
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    print(cost[i])

predict = np.dot(x,w)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')

#Result
x1 = 50
y1 = w[0] + w[1]*x1


plt.show()