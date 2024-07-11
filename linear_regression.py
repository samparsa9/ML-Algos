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
        
        self.x_features = None
        self.z_features = None
        self.y_labels = None
        
        self.y_preds = None   

    # This function is essentially our model, which is just the equation for a line y = mx + b 
    # y (predictions) = y-intercept + slope x + slope z
    # y (predictions) = y-intercept (b) + wx + nz
    def f(self,x_features, z_features, w, n, b): 
        return (x_features * w) + (z_features * n) + b #y

    # This is our cost function, which we have chosen to be Mean Squarred Error (Sum of Squarred Error leads to exploding gradients)
    def loss_function(self, x_features, z_features, y_labels, w, n, b):
        self.y_preds = self.f(x_features, z_features, w, n, b) # y_hat is a list of our models predictions; produces our y values to plot our line of best fit
        return np.mean((y_labels - self.y_preds) ** 2) # Returning the MSE for these current parameters


    # While either of our partial derivatives are greater than 0.1, keep running gradient descent
    def fit(self,x_features,z_features, y_labels):
        # Storing the data passed in as class variables to be used later on
        self.x_features = x_features
        self.z_features = z_features
        self.y_labels = y_labels
        #self.params = (self.params.append(random.randint(10)) for x in features_matrix)
        # This is our epsilon/small value to be used in calculating our partial derivatives
        h = 1e-5
        # Our model is currently not optimized
        optimized = False
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # While its not optimized, run gradient descent
        while not optimized:
            # The formula for the partial derivative of our MSE loss function with respect to our first parameter, w, 
            # representing the slope, can be seen as follows
            dw = (self.loss_function(x_features,z_features,y_labels,self.w+h,self.n,self.b) - self.loss_function(x_features,z_features,y_labels,self.w-h,self.n,self.b)) / (2*h)
            #print(f"New dw: {dw}")
            # The formula for the partial derivative of our MSE loss function with respect to our second parameter, n, 
            # representing the slope, can be seen as follows
            dn = (self.loss_function(x_features,z_features,y_labels,self.w,self.n+h,self.b) - self.loss_function(x_features,z_features,y_labels,self.w,self.n-h,self.b)) / (2*h)
            # The formula for the partial derivative of our MSE loss function with respect to our third parameter, b, 
            # representing the intercept, can be seen as follows
            db = (self.loss_function(x_features,z_features,y_labels,self.w,self.n,self.b+h ) - self.loss_function(x_features,z_features,y_labels,self.w,self.n,self.b-h)) / (2*h)
            #print(f"New db: {db}")
            # Using our learning rate, we will add a tiny bit to our slope in the oppsite signed direction of our partial derivative
            temp_w = self.w - ((self.learning_rate * dw))
            temp_n = self.n - ((self.learning_rate * dn))
            # And do the same for our intercept
            temp_b = self.b - ((self.learning_rate * db))
            
            # Then update the entire class' w parameter
            self.w = temp_w
            self.n = temp_n
            # And the entire class' b parameter
            self.b = temp_b
            
            
            self.y_preds = self.f(x_features, z_features, self.w, self.n, self.b)

            # Visualization of Algorithm
            ax.cla()
            ax.scatter(self.x_features, self.z_features, self.y_labels, color='red', label='Actual')
            ax.plot_trisurf(self.x_features, self.z_features, self.y_preds, color='blue', alpha=0.5, label='Predicted')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            plt.draw()
            plt.pause(0.1)

            # If both of these partial derivatives are less than 0.1, we have completed gradient descent
            if abs(dw) < 0.1 and abs(db) < 0.1 and abs(dn) < 0.1:
                # Our model is now optimized
                optimized = True
                print(f"dw: {dw}\tdn: {dn}\tdb: {db}")
                
        
    # Get our new predictions array by inputting our previous x data into our now optimized model (the line of best fit)
    def predict(self, x_features, z_features):
        self.y_preds = self.f(x_features, z_features, self.w, self.n, self.b)
        return self.y_preds
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_features, self.z_features, self.y_labels, color='red', label='Actual')
        ax.plot_trisurf(self.x_features, self.z_features, self.y_preds, color='blue', alpha=0.5, label='Predicted')
        ax.set_xlabel('X Feature')
        ax.set_ylabel('Z Feature')
        ax.set_zlabel('Y Labels')
        #plt.legend()
        plt.show()
        # # Plot our actual x and y points
        # plt.plot(self.x_features, self.y_labels, 'o')
        # # Plot our models predictions using the x values
        # plt.plot(self.x_features, self.y_preds)
        # plt.plot(self.z_features)
        # plt.show()


example_dataset_1 = [[0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2], #x_features
                     [1.2, 1.0, 2.0, 4.0, 3.5, 6.0, 4.7, 4.4, 7.0], #z_features
                     [1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0], #y_labels
                     ]

data = np.array(example_dataset_1)


model = LinearRegression()
start = datetime.datetime.now()
model.fit(data[0], data[1], data[-1])
end = datetime.datetime.now()
print(f"time elapsed: {end - start}")
model.predict(data[0], data[1])
model.plot()



    



    