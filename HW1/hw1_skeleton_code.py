import pandas as pd
import logging
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

### Assignment Owner: Tian Wang

#######################################
####Q2.1: Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    train_min = np.amin(train,axis = 0)
    train_X = np.subtract(train,train_min)
    train_range = np.amax(train_X,axis = 0)
    train_rescaled = train_X/train_range

    test_X = np.subtract(test,train_min)
    test_rescaled = test_X/train_range

    return train_rescaled, test_rescaled

########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
   
    #TODO
    h_theta = np.dot(X,theta)
    loss = np.dot(np.power(np.subtract(h_theta, y),2),np.ones(X.shape[0]))/(2*X.shape[0])
    return loss

########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    h_theta = np.dot(X,theta)
    loss_gradient = np.dot(np.subtract(h_theta,y),X)/X.shape[0]
    return loss_gradient
    
       
        
###########################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    basis_vectors = np.identity(X.shape[1])
    directional_change = epsilon * basis_vectors
    
    for i in range(num_features):
        approx_grad[i] = (compute_square_loss(X,y,theta + directional_change[i]) - compute_square_loss(X,y,theta - directional_change[i]))/(2*epsilon)
   
    eucledian_dist = np.linalg.norm(approx_grad-true_gradient)
    if (eucledian_dist <= tolerance):
        return True
    else:
        return False

#################################################
###Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    basis_vectors = np.identity(X.shape[1])
    directional_change = epsilon * basis_vectors
    
    for i in range(num_features):
        approx_grad[i] = (objective_func(X,y,theta + directional_change[i]) - objective_func(X,y,theta - directional_change[i]))/(2*epsilon)
        
    eucledian_dist = np.linalg.norm(approx_grad-true_gradient)
    if (eucledian_dist <= tolerance):
        return True
    else:
        return False

####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    #TODO
    for i in range(num_iter):
        theta_hist[i] = theta
        loss_hist[i] = compute_square_loss(X,y,theta)

        gradient = compute_square_loss_gradient(X,y,theta)
        theta = theta - (alpha * gradient)

        if not grad_checker(X, y, theta):
            print "grad check failed  " , i
            return theta_hist, loss_hist

     
    loss_plot = plt.plot(loss_hist)
    plt.title('Loss as a function of number of steps')
    plt.ylabel('loss')
    plt.xlabel('No.of Steps')
    file_name = 'loss_'+str(alpha)+'.png'
    plt.savefig(file_name)
    return theta_hist, loss_hist

####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
    


###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    reg_loss_gradient = compute_square_loss_gradient(X,y,theta) + (2*lambda_reg*theta)
    return reg_loss_gradient

###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha, lambda_reg, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO

    for i in range(num_iter):
        theta_hist[i] = theta
        loss_hist[i] = compute_regularized_square_loss(X,y,theta,lambda_reg)

        gradient = compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        theta = theta - (alpha * gradient)

    return theta_hist,loss_hist


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    return compute_square_loss(X,y,theta) + lambda_reg*(np.dot(theta.T,theta))
    
   
#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss
 
def vis_bgd(X_train, X_test, y_train, y_test):
        
    lambda_range = np.arange(1,5,1)
    square_loss_train = np.zeros(lambda_range.shape[0])
    square_loss_test = np.zeros(lambda_range.shape[0])
    k = 0

    for i in lambda_range:
        train_theta,train_loss = regularized_grad_descent(X_train,y_train,0.1,i)
        test_theta,test_loss = regularized_grad_descent(X_test,y_test,0.1,i)
        
        min_loss_index_train = np.argmin(train_loss)
        min_loss_theta_train = train_theta[min_loss_index_train]
        print min_loss_theta_train
       
        """
        min_loss_index_test = np.argmin(test_loss)
        min_loss_theta_test = test_theta[min_loss_index_test]
        print min_loss_index_test
        """
        square_loss_train[k] =compute_square_loss(X_train,y_train,min_loss_theta_train)
        #square_loss_test[k] = compute_square_loss(X_test,y_test,min_loss_theta_test)

        k = k + 1

    print square_loss_train
    print square_loss_test

        


#############################################
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    #TODO
    
    permutation = np.random.permutation(X.shape[0])
    
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    
    for i in range(num_iter):
        for j in range(num_instances):
            theta_hist[i,j] = theta
            loss_hist[i,j] = np.power(np.subtract(np.dot(X_shuffled[j],theta),y_shuffled[j]),2) + lambda_reg*(np.dot(theta.T,theta))

            gradient = np.dot(np.subtract(np.dot(theta,X_shuffled[j]),y_shuffled[j]),X_shuffled[j]) + 2*lambda_reg*theta
            theta = theta - (alpha * gradient)
        
    return theta_hist,loss_hist
################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

def main():
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)
    
    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term
    
    # TODO
    loss = compute_square_loss(X_train,y_train,np.ones(X_train.shape[1]))
    loss_gradient = compute_square_loss_gradient(X_train,y_train,np.ones(X_train.shape[1]))

    grad_check = grad_checker(X_train,y_train,np.ones(X_train.shape[1]))
    print grad_check
    
    vis_bgd(X_train, X_test, y_train, y_test)
    #theta_hist, loss_hist = batch_grad_descent(X_train,y_train,0.01)
    #theta_hist, loss_hist = batch_grad_descent(X_train,y_train,0.05)
    #theta_hist, loss_hist = batch_grad_descent(X_train,y_train,0.1)
    #theta_hist, loss_hist = batch_grad_descent(X_train,y_train,0.5)    
    
    #theta_hist, loss_hist = regularized_grad_descent(X_train,y_train)
    
    #theta_hist_sgd, loss_hist_sgd = stochastic_grad_descent(X_train,y_train)
    
    #print theta_hist_sgd
    #print loss_hist_sgd

if __name__ == "__main__":
    main()
