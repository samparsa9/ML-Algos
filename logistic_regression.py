import matplotlib.pyplot as plt
import numpy as np
import datetime
from numpy import random

class LogisticRegression:
    #USES NATURAL LOG
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.b = 0
        self.params = []
        self.model_refresh = 0.01
        self.features_matrix = None
        self.y_labels = None
        self.floating_y_preds = None
        self.y_preds = None

    def f(self, features_matrix, params, b):
        bruh = np.dot(features_matrix.T, params) + b
        # print(features_matrix)
        print(f"parameters are: {params}")
        print(bruh)
        return bruh
    
    def cast_to_p(self, unscaled_pred):
        #Ln(p/1-p) = 2
        #e^(ln(p/1-p)) = e^(2)
        # p / (1-p) = e^(2)
        # p = 7.4(1 - p)
        # p = 7.4 - 7.4p
        # 8.4p = 7.4
        # p = 0.88
        num_to_e = np.exp(unscaled_pred)
        
        p = (num_to_e) / (num_to_e+1)
        return p

    #Log Loss function
    #self, features_matrix, y_labels, params, b
    def loss_function(self, features_matrix, y_labels, params, b): #y_labels will be a list of [0 or 1], 
        self.floating_y_preds = np.array([self.cast_to_p(unscaled_p) for unscaled_p in self.f(features_matrix, params, b)])    
        #p: value between 0 and 1 (0.5 is midline)
        #when u plug in p, you get value between (-inf, inf), -3/3 is 95% confidence
        return np.sum(((-1 * y_labels) * np.log(self.floating_y_preds)) - ((1-y_labels) * np.log(1-self.floating_y_preds)))

    def fit(self,features_matrix, y_labels):
        # Storing the data passed in as class variables to be used later on
        self.features_matrix = features_matrix # [[features_1],[features_2], ... [y_labels]]
        self.y_labels = y_labels
        self.params = [random.randint(10) for _ in range(len(features_matrix))]
        print("in fit function")
        print(self.features_matrix)
        print(self.params)
        # This is our epsilon/small value to be used in calculating our partial derivatives
        h = 1e-5
        # Our model is currently not optimized
        optimized = False
        plt.ion()
        fig = plt.figure()
        if len(self.params) == 2:
            ax = fig.add_subplot(111, projection='3d')
        if len(self.params) == 1:
            ax = fig.add_subplot(111)
        # While its not optimized, run gradient descent
        while not optimized:
            partial_derivatives = []
            temp_params = []
            #print(self.params)
            for i in range(len(self.params)): #  self.params [w, b, z]
                paramsplush = self.params[:i] + [self.params[i] + h] + self.params[i+1:]
                paramsminush = self.params[:i] + [self.params[i] - h] + self.params[i+1:]
                dparam = (self.loss_function(features_matrix,y_labels,paramsplush,self.b)
                        - self.loss_function(features_matrix,y_labels,paramsminush,self.b)) / (2*h)
                partial_derivatives.append(dparam)
                self.params[i] = self.params[i] - ((self.learning_rate * dparam))
                print(f"Partial Derivative: {dparam}")
            
            db = (self.loss_function(features_matrix,y_labels,self.params,self.b+h ) - self.loss_function(features_matrix,y_labels,self.params,self.b-h)) / (2*h)
            
            self.b = self.b - ((self.learning_rate * db))
                       
            
            self.y_preds = (self.f(self.features_matrix, self.params, self.b))

            self.live_plotting(ax)

            # If both of these partial derivatives are less than 0.1, we have completed gradient descent
            if all(abs(pd) < 0.01 for pd in partial_derivatives):
                optimized = True
                print(f"partial derivatives: {partial_derivatives}")

# Get our new predictions array by inputting our previous x data into our now optimized model (the line of best fit)
    def predict(self, features_matrix):
        self.y_preds = np.heaviside(self.f(features_matrix, self.params, self.b), 0.5)
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

    def live_plotting(self, ax):
        if len(self.params) == 2:
                # Visualization of Algorithm
                ax.cla()
                ax.scatter(self.features_matrix[0],self.features_matrix[1], self.y_labels, color='red', label='Actual')
                ax.plot_trisurf(self.features_matrix[0], self.features_matrix[1], self.y_preds, color='blue', alpha=0.5, label='Predicted')
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_zlabel('Y')
                plt.draw()
                plt.pause(self.model_refresh)
        if len(self.params) == 1:
            ax.cla()
            ax.plot(self.features_matrix[0], self.y_labels, 'o')
            # Plot our models predictions using the x values
            plt.plot(self.features_matrix[0], self.y_preds)
            plt.draw()
            plt.pause(self.model_refresh)


def main():
    data1 = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9], # Weight of a mouse in pounds
        [0, 0, 0, 0, 1, 0, 1, 1, 1], # 0 = Not obese, 1 = Obese
    ]
    x_features = np.array(data1[0:-1])
    y_label = np.array(data1[-1])

    model = LogisticRegression()
    model.fit(x_features, y_label)
    print(model.predict(x_features))


    
    
if __name__ == "__main__":
    main()
