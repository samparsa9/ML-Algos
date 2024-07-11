import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import scipy as sc
from scipy import optimize
from scipy.misc import derivative
from findiff import FinDiff
import time as tm
from numpy import random

class LinearRegression:
    def __init__(self, learning_rate=0.01): 
        # Initializing class variables
        self.learning_rate = learning_rate
        # Initializing our parameters to start off as random numbers between 1 and 10
        self.w = random.randint(10)
        self.b = random.randint(10)
        self.x = None
        self.y = None
        self.predictions = None   

    # This function is essentially our model, which is just the equation for a line y = mx + b
    def f(self,x, w, b): 
        return (x * w) + b

    # This is our cost function, which we have chosen to be Mean Squarred Error (Sum of Squarred Error leads to exploding gradients)
    def loss_function(self, x, y, w, b):
        y_hat = self.f(x, w, b) # y_hat are our models predictions, it produces our y values to plot our line of best fit
        return np.mean((y - y_hat) ** 2) # Returining the MSE for these current parameters


    # While either of our partial derivatives are greater than 0.1, keep running gradient descent
    def fit(self,x,y):
        # Storing the data passed in as class variables to be used later on
        self.x = x
        self.y = y
        # This is our epsilon/small value to be used in calculating our partial derivatives
        h = 1e-5
        # Our model is currently not optimized
        optimized = False
        # plt.ion()
        # fig, ax = plt.subplots()
        # While its not optimized, run gradient descent
        while not optimized:
            # The formula for the partial derivative of our MSE loss function with respect to our first parameter, w, 
            # representing the slope, can be seen as follows
            dw = (self.loss_function(x,y,self.w+h,self.b) - self.loss_function(self.x,self.y, self.w-h, self.b)) / (2*h)
            print(f"New dw: {dw}")
            # The formula for the partial derivative of our MSE loss function with respect to our second parameter, b, 
            # representing the intercept, can be seen as follows
            db = (self.loss_function(x,y,self.w,self.b+h ) - self.loss_function(x,y,self.w, self.b-h)) / (2*h)
            print(f"New db: {db}")
            # Using our learning rate, we will add a tiny bit to our slope in the oppsite signed direction of our partial derivative
            # little_w_to_add = 0
            # if dw > dw/100: little_w_to_add += dw/100
            temp_w = self.w + (-1*((self.learning_rate) * dw))
            # And do the same for our intercept
            # little_b_to_add = 0
            # if db > db/100: little_b_to_add += db/100
            temp_b = self.b + (-1*((self.learning_rate) * db))
            # Then update the entire class' w parameter
            self.w = temp_w
            # And the entire class' b parameter
            self.b = temp_b
            # print(f"new b: {self.b}")
            # self.predictions = self.f(x, self.w, self.b)
            # ax.cla()  # Clear the previous plot
            # ax.plot(self.x, self.y, 'o', label="Data Points")
            # ax.plot(self.x, self.predictions, label="Fitted Line")
            # ax.legend()
            # plt.draw()  # Update the plot
            # plt.pause(0.05)  # Pause for a short interval to allow the plot to update
            # If both of these partial derivatives are less than 0.1, we have completed gradient descent
            if abs(dw) < 0.1 and abs(db) < 0.1:
                # Our model is now optimized
                optimized = True
        
    # Get our new predictions array by inputting our previous x data into our now optimized model (the line of best fit)
    def predict(self, x):
        self.predictions = self.f(x, self.w, self.b)
        return self.predictions
    
    def plot(self):
        # Plot our actual x and y points
        plt.plot(self.x, self.y, 'o')
        # Plot our models predictions using the x values
        plt.plot(self.x, self.predictions)
        plt.show()


data = [[0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2],[1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0]]
#y = np.array([1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0])
#y = 2 * x + np.random.normal(0, 1, 100)

data = np.array(data)
model = LinearRegression()
model.fit(data[0], data[1])
print(model.predict(data[0]))



    



    