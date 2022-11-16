import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Tim Min f(x) = x^2 + 2*x + 5

def derivative(x):
    return 2*x + 2

def Min(x0, learning_rate, iteration):
    a = []
    for i in range(iteration):
        x0 -= learning_rate*derivative(x0)
        a.append(x0)
    return x0

a = Min(5,0.01,600)
print(a)


