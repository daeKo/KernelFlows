# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:07:58 2020

@author: matth
"""


import numpy as np
import math

from kernel_functions import kernels_dic
from nabla_functions import nabla_dic


#%% 
        
"""
# Instructions:
    
# 1) Initialize a instance. 
# Kernel name is a string, with the name of kernel 
# (see dictionnary below). For a custome kernel, create the function and add 
# the name to the dictionnary.
# Mu is a np array with all the parameters of your kernel (depends on the kernel).
        
KF = KernelFlowsP(kernel_name, mu)

# 2) To learn the best parameters, call the fit function

# Mandatory arguments:
# data: np array with both X and Y (independent and dependent variables). 

# iterations: integer

# Optional arguments
# batch_size: either an integer, a float between 0 and 1 or False. If an integer,
# this determines the batch size. If between 0 and 1, the batch size is the proportion
# of the data set (ex: 0.25 of the data set is used per batch). If False (default),
# each batch is the full data set (not recommended except for small data sets).

# optimizer: string. Either "SGD" or "Nesterov". Determines the update rule. Default
# is "SGD".

# learning_rate: small float. Default is 0.1

# beta: float beween 0 and 1. Only used for the Nesterov update rule. Default is 0.9.

# show_it: how often to show the number of iterations done. Default is 100. 

# regu_lambda: small float. Determines the amount of regularization. Default is 0.0001.  


mu_pred = KF.fit(data, iterations)

# 3) To generate a prediction, call the predict function
    
# test: a np array. The dimension should be d-1 compared to the data (i.e. should 
# have all independent variables, but not the dependent varaible Y). Returns 
# 1 dimensional array.  
pred = KF.predict(test)

# IMPORTANT. If your kernel is a standard kernel availabel from sklearn.kernel_ridge.KernelRidge
# it is strongly recommended you use this instead, as it is more efficient. Note 
# for the RBF kernel, gamma = 1/(2*sigma**2)

# For use outside of this file, make sure the file is in the same directory and 
# use "from KF_parametric_class import KernelFlowsP" and do everything as above
"""


#%%
    
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices

# Splits the data into the target and predictor variables.
def split(data):
    X = data[:, :-1]
    Y = data[:, -1]
    
    return X, Y


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]

    
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    
    t_matrix = kernel(X_test, X_train, param)
    
    prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train))
    
    return prediction

def replace_nan(array):
    for i in range(array.shape[0]):
        if math.isnan(array[i]) == True:
            print("Found nan value, replacing by 0")
            array[i] = 0
    return array

def sample_size_linear(iterations, range_tuple):
    
    return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]
            
#%% Rho function

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi



def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF"):
    kernel = kernels_dic[kernel_keyword]
    # norm_square = np.sum(np.square(matrix_data), axis = 1)
    # norm_square = np.reshape(norm_square, (-1,1))
    # inner_matrix = np.matmul(matrix_data, np.transpose(matrix_data))
    
    # norm_diff = -2 * inner_matrix + norm_square + np.transpose(norm_square)
    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = 0.000001
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    
    top = np.dot(Y_sample, np.matmul(inverse_sample, Y_sample))
    bottom = np.dot(Y_data, np.matmul(inverse_data, Y_data))

    return 1 - top/bottom

def l2(parameters, matrix_data, Y, batch_indices, sample_indices, kernel_keyword = "RBF"):
    X_sample = matrix_data[sample_indices]
    Y_sample = Y[sample_indices]
    
    not_sample = [x for x in batch_indices not in sample_indices]
    X_not_sample = matrix_data[not_sample]
    Y_not_sample = Y[not_sample]
    prediction = kernel_regression(X_sample, X_not_sample, Y_sample, kernel_keyword)
    
    return np.dot(Y_not_sample - prediction, Y_not_sample- prediction)

#%% Grad functions

""" We define the gradient calculator function.Like rho, the gradient 
calculator function accesses the gradfunctions via a keyword"""

# Gradient calculator function. Returns an array
def frechet_grad(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", regu_lambda = 0.000001):
    #Getting the derivative matrix and the theta matrix
    nabla_and_theta = nabla_dic[kernel_keyword]
    nabla_matrix, theta = nabla_and_theta(X_data, parameters)


    # Rho
    pi = pi_matrix(sample_indices, (sample_indices.shape[0],X_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(theta, np.transpose(pi)))
    Y_sample = Y_data[sample_indices]
    
    lambda_term = regu_lambda
    inverse_data = np.linalg.inv(theta + lambda_term * np.identity(theta.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    
    top = np.dot(Y_sample, np.matmul(inverse_sample, Y_sample))
    bottom = np.dot(Y_data, np.matmul(inverse_data, Y_data))
    rho = 1 - top/bottom
    
    # Computing Y_hat and Z_hat (see paper)
    y_hat = np.expand_dims(np.matmul(inverse_data, Y_data), 1)
    z_hat = np.expand_dims(np.matmul(np.transpose(pi), np.matmul(inverse_sample, np.matmul(pi, Y_data))),1)

    gradient = ((1-rho)*np.matmul(np.transpose(y_hat), np.matmul(nabla_matrix,y_hat)) - np.matmul(np.transpose(z_hat), np.matmul(nabla_matrix, z_hat)))
    gradient = -gradient/bottom
    gradient = np.squeeze(gradient)
    #print(gradient, rho)
    return gradient, rho


#%% The class version of KF
    
class KernelFlowsP():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        
        self.LR = 0.1
        self.beta = 0.9
        self.regu_lambda = 0.0001
    
    def get_hist(self):
        return self.param_hist, self.gradients, self.rho_values
        
    
    def save_model(self):
        np.save("param_hist", self.param_hist)
        np.save("gradients", self.gradients)
        np.save("rho_values", self.rho_values)
        
    def get_parameters(self):
        return self.parameters
    
    def set_LR(self, value):
        self.LR = value
        
    def set_beta(self, value):
        self.beta = value
    def set_train(self, train):
        self.train = train
        
    
    def fit(self, X, Y, iterations, batch_size = False, optimizer = "SGD", 
            learning_rate = 0.1, beta = 0.9, show_it = 100, regu_lambda = 0.000001, 
            adaptive_size = False, adaptive_range = (), proportion = 0.5, reduction_constant = 0.0):            
        #print(self.parameters)
        self.set_LR(learning_rate)
        self.set_beta(beta)
        self.regu_lambda = regu_lambda
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        momentum = np.zeros(self.parameters.shape, dtype = "float")
        
        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        for i in range(iterations):
            if i % show_it == 0:
                print("Iteration ", i)
            
            if adaptive_size == "Linear":
                sample_size = sample_size_array[i]
                
            elif adaptive_size == "Dynamic" and adaptive_counter == 100:
                if adaptive_mean != 0:
                    change = np.mean(rho_100) - adaptive_mean 
                else:
                    change = 0
                adaptive_mean = np.mean(rho_100)
                rho_100 = []
                sample_size += change - reduction_constant
                adaptive_counter= 0
                
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size, sample_proportion = sample_size)
            X_data = X[batch_indices]
            Y_data = Y[batch_indices]
            

                
            # Changes parameters according to SGD rules
            if optimizer == "SGD":
                grad_mu, rho = frechet_grad(self.parameters, X_data, Y_data, 
                                           sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1.01 or rho < -0.1:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    self.parameters -= learning_rate * grad_mu
                    
            
            # Changes parameters according to Nesterov Momentum rules     
            elif optimizer == "Nesterov":
                grad_mu, rho = frechet_grad(self.parameters, X_data, Y_data, 
                                           sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1.01 or rho < -0.1:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    momentum = beta * momentum + grad_mu
                    self.parameters -= learning_rate * momentum
                
            else:
                print("Error optimizer, name not recognized")
            
            # Update history 
            self.para_hist.append(np.copy(self.parameters))
            self.rho_values.append(rho)
            self.grad_hist.append(np.copy(grad_mu))
            
            rho_100.append(rho)
            adaptive_counter +=1
                
            
        # Convert all the lists to np arrays
        self.para_hist = np.array(self.para_hist) 
        self.rho_values = np.array(self.rho_values)
        self.grad_hist = np.array(self.grad_hist)
                
        return self.parameters
    
    def predict(self,test, regu_lambda = 0.000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_regression(X_train, test, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

#%%
        
if __name__ == "__main__":
    # Generating data according to RBF kernel, true gamma is 0.1
    from autograd.numpy.random import uniform
    def data_set_RBF(dimensions, mu_correct):
        # a = normal(scale=0.2, size = size)
        values = uniform(-10, 10, dimensions)
        b = []
        for element in values:
            b.append( np.exp(-np.linalg.norm(element)**2 /(2*mu_correct[0]**2)))
        b = np.array(b) #+ normal(0, 0.25)
        return b, values

    mu_correct = np.array([10.0])
    Y, X = data_set_RBF((100, 1), mu_correct)
    data_set = np.concatenate((X,np.expand_dims(Y, 1)), axis = 1)
    
    mu_1 = np.array([5.0])
    K = KernelFlowsP("RBF", mu_1)
    mu_pred = K.fit(X, Y, 10000, optimizer = "Nesterov",  batch_size = 50, show_it = 5000)
    print(mu_pred)
    
    mu_2 = np.array([15.0])
    K = KernelFlowsP("RBF", mu_2)
    mu_pred = K.fit(X, Y, 10000, optimizer = "Nesterov", batch_size = 50, show_it = 5000)
    print(mu_pred)

#%%

# pred = K.predict(X[:50])