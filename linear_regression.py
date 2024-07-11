import matplotlib.pyplot as plt
import numpy as np
import datetime
from numpy import random

class LinearRegression:
    def __init__(self, learning_rate=0.02): 
        # Initializing class variables
        self.learning_rate = learning_rate
        # Initializing our parameters to start off as random numbers between 1 and 10
        self.w = random.randint(10)
        self.b = random.randint(10)
        self.n = random.randint(10)
        self.params = []
        
        self.features_matrix = None
        self.y_labels = None
        
        self.y_preds = None   

    # This function is essentially our model, which is just the equation for a line y = mx + b 
    def f(self,features_matrix, params, b): 
        return np.dot(features_matrix.T, params) + b #y

    # This is our cost function, which we have chosen to be Mean Squarred Error (Sum of Squarred Error leads to exploding gradients)
    def loss_function(self, features_matrix, y_labels, params, b):
        self.y_preds = self.f(features_matrix, params, b) # y_hat is a list of our models predictions; produces our y values to plot our line of best fit
        return np.mean((y_labels - self.y_preds) ** 2) # Returning the MSE for these current parameters


    # While either of our partial derivatives are greater than 0.1, keep running gradient descent
    def fit(self,features_matrix, y_labels):
        # Storing the data passed in as class variables to be used later on
        self.features_matrix = features_matrix
        self.y_labels = y_labels
        self.params = [random.randint(10) for _ in range(len(features_matrix))]
        # This is our epsilon/small value to be used in calculating our partial derivatives
        h = 1e-5
        # Our model is currently not optimized
        optimized = False
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # While its not optimized, run gradient descent
        while not optimized:
            partial_derivatives = []
            temp_params = []
            #print(self.params)
            for i in range(len(self.params)):
                paramsplush = self.params[:i] + [self.params[i] + h] + self.params[i+1:]
                paramsminush = self.params[:i] + [self.params[i] - h] + self.params[i+1:]
                dparam = (self.loss_function(features_matrix,y_labels,paramsplush,self.b)
                        - self.loss_function(features_matrix,y_labels,paramsminush,self.b)) / (2*h)
                partial_derivatives.append(dparam)
                temp_params.append(self.params[i] - ((self.learning_rate * dparam)))
                self.params[i] = temp_params[i]
                print(f"Derivative: {self.params[i]}")
            # The formula for the partial derivative of our MSE loss function with respect to our first parameter, w, 
            # representing the slope, can be seen as follows
            #dw = (self.loss_function(features_matrix,y_labels,self.w+h,self.n,self.b) - self.loss_function(features_matrix,y_labels,self.w-h,self.n,self.b)) / (2*h)
            #print(f"New dw: {dw}")
            # The formula for the partial derivative of our MSE loss function with respect to our second parameter, n, 
            # representing the slope, can be seen as follows
            #dn = (self.loss_function(features_matrix,y_labels,self.w,self.n+h,self.b) - self.loss_function(features_matrix,y_labels,self.w,self.n-h,self.b)) / (2*h)
            # The formula for the partial derivative of our MSE loss function with respect to our third parameter, b, 
            # representing the intercept, can be seen as follows
            db = (self.loss_function(features_matrix,y_labels,self.params,self.b+h ) - self.loss_function(features_matrix,y_labels,self.params,self.b-h)) / (2*h)
            #print(f"New db: {db}")
            # Using our learning rate, we will add a tiny bit to our slope in the oppsite signed direction of our partial derivative
            #temp_w = self.w - ((self.learning_rate * dw))
            #temp_n = self.n - ((self.learning_rate * dn))
            # And do the same for our intercept
            temp_b = self.b - ((self.learning_rate * db))
            
            # Then update the entire class' w parameter
            #self.w = temp_w
            #elf.n = temp_n
            # And the entire class' b parameter
            self.b = temp_b
           
            
            self.y_preds = self.f(self.features_matrix, self.params, self.b)

            # Visualization of Algorithm
            ax.cla()
            ax.scatter(self.features_matrix[0],self.features_matrix[1], self.y_labels, color='red', label='Actual')
            ax.plot_trisurf(self.features_matrix[0], self.features_matrix[1], self.y_preds, color='blue', alpha=0.5, label='Predicted')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            plt.draw()
            plt.pause(0.1)

            # If both of these partial derivatives are less than 0.1, we have completed gradient descent
            if all(abs(pd) < 0.01 for pd in partial_derivatives):
                optimized = True
                print(f"partial derivatives: {partial_derivatives}")
                
        
    # Get our new predictions array by inputting our previous x data into our now optimized model (the line of best fit)
    def predict(self, features_matrix):
        self.y_preds = self.f(features_matrix, self.params, self.b)
        return self.y_preds
    
    def plot(self):
        if len(self.params) == 1:
            # Plot our actual x and y points
            plt.plot(self.features_matrix[0], self.y_labels, 'o')
            # Plot our models predictions using the x values
            plt.plot(self.features_matrix[0], self.y_preds)
            plt.show()
        elif len(self.params) == 2:
            print('Here')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.features_matrix[0],self.features_matrix[1], self.y_labels, color='red', label='Actual')
            ax.plot_trisurf(self.features_matrix[0],self.features_matrix[1], self.y_preds, color='blue', alpha=0.5, label='Predicted')
            ax.set_xlabel('X Feature')
            ax.set_ylabel('Z Feature')
            ax.set_zlabel('Y Labels')
            #plt.legend()
            plt.show()
            print("here")
        else:
            print(f"Cannot display graph of {len(self.params) + 1} Dimensions")
        



data1 = [[0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2],[1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0]]
data2 = [
    [0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2, 6.5, 6.8, 7.0, 7.3, 7.6, 8.0, 8.4, 8.8, 9.0, 9.3, 9.7], # x_features
    [1.2, 1.0, 2.0, 4.0, 3.5, 6.0, 4.7, 4.4, 7.0, 7.4, 7.8, 8.0, 8.3, 8.6, 9.0, 9.4, 9.8, 10.0, 10.3, 10.7], # z_features
    [1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0, 6.3, 6.6, 7.0, 7.3, 7.6, 8.0, 8.3, 8.7, 9.0, 9.4, 9.8], # y_labels
]

data3 = [
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5], # x_features
    [0.7, 1.3, 1.9, 2.5, 3.1, 3.7, 4.3, 4.9, 5.5, 6.1, 6.7, 7.3, 7.9, 8.5, 9.1, 9.7], # z_features
    [0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 6.6, 7.2, 7.8, 8.4, 9.0], # y_labels
]



x_features = np.array(data2[0:-1])
y_label = np.array(data2[-1])

model = LinearRegression()
start = datetime.datetime.now()
model.fit(x_features, y_label)
#print(f"parameters {model.params}")
end = datetime.datetime.now()
#print(f"time elapsed: {end - start}")
#print(model.predict(x_features))
#model.plot()



    



    