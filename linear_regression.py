import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#x = np.linspace(0, 10, 100)
data = [[0.8, 1.5, 2.3, 3.4, 4.4, 5.2, 5.4, 5.7, 6.2],[1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0]]
#y = np.array([1.2, 1.5, 2.5, 3.1, 3.3, 3.8, 4.8, 5.5, 6.0])
#y = 2 * x + np.random.normal(0, 1, 100)

data = np.array(data)

def LinearRegression(training_data, num_features=1): # num_features will specify the shape of the input data
    #data is a list of dimensions which has a vector each
    #ex) x_train and y_train are x and y values that correspond with each other, both in the 'data' list    

    #the last list in data will be the label vector (what we are predicting)
    #any vectors preceding will
    x_train = np.array(training_data[0])
    y_train = np.array(training_data[-1])
    #print(xpoints)
    #print(ypoints)
    
    #STEPS:
    #1) Plot Data

    plt.plot(x_train, y_train, 'o')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    

    w = 0
    b = 3
    def fit_line(x_train, w, b):
        y_hat = np.multiply(x_train, w)
        y_hat = np.add(y_hat,b)
        return y_hat
    
    y_hat = fit_line(x_train, w, b)
    print(y_hat)
    print(y_hat.shape)
    plt.plot(x_train, y_hat, '')
    

    plt.show()
    
    return (w,b)




LinearRegression(data)
#print(LinearRegression(data))

    



    