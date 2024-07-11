import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import scipy as sc
from scipy import optimize
from scipy.misc import derivative
from findiff import FinDiff
import time as tm

class LinearRegression: # num_features will specify the shape of the input data
    def __init__(self, num_features=1, learning_rate=0.01, w=-0, b=0):
        self.learning_rate = learning_rate
        self.w = w
        self.b = b
        self.x = None
        self.y = None
        self.predictions = None
    #data is a list of dimensions which has a vector each
    #ex) x_train and y_train are x and y values that correspond with each other, both in the 'data' list    

    #the last list in data will be the label vector (what we are predicting)
    #any vectors preceding will
    

    # Function/Equation for a line
    def f(self,x, w, b): 
        return (x * w) + b

    # This is our cost function that we need to take partial derivates of parameters with respect to
    def loss_function(self, x, y, w, b):
        y_hat = self.f(x, w, b) # y_hat is our predictions, it produces our y values to plot our line of best fit
        return np.mean((y - y_hat) ** 2)


    # While either of our partial derivatives are greater than 0.01, keep running gradient descent
    def fit(self,x,y):
        self.x = x
        self.y = y
        h = 1e-5
        optimized = False
        while not optimized:
            dw = (self.loss_function(x,y,self.w+h,self.b) - self.loss_function(self.x,self.y, self.w-h, self.b)) / (2*h)
            print(f"New dw: {dw}")
            tm.sleep(0.1)
            db = (self.loss_function(x,y,self.w,self.b+h ) - self.loss_function(x,y,self.w, self.b-h)) / (2*h)
            print(f"New db: {db}")
            tm.sleep(0.1)
            temp_w = self.w + (-1*(self.learning_rate * dw))
            temp_b = self.b + (-1*(self.learning_rate * db))
            self.w = temp_w
            # print(f"new w: {self.w}")
            self.b = temp_b
            # print(f"new b: {self.b}")
            if abs(dw) < 0.1 and abs(db) < 0.1:
                optimized = True
    
        self.predictions = self.f(x, self.w, self.b)


    
    def plot(self):
        plt.plot(self.x, self.y, 'o')
        plt.plot(self.x, self.predictions)
        #plt.xlim(0, 7)
        #plt.ylim(0, 7)
        plt.show()


data = [[0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2],[1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0]]
#y = np.array([1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0])
#y = 2 * x + np.random.normal(0, 1, 100)

data = np.array(data)
model = LinearRegression()
model.fit(data[0], data[1])
print(model.y)
print(model.predictions)
model.plot()

#print(LinearRegression(data))

    



    