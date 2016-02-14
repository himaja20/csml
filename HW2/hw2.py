import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math


def ridge(X,y,theta,Lambda):
	def ridge_obj(theta):
	  return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*N) + Lambda*(np.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(X,y,theta):
	return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*N)

def coordinate_descent(X,y,lamda,w):
	print 'lamda in coordinte descent function  ' , lamda	
	loss = float("inf")
	new_loss = 0
	while(True):
		if (loss - new_loss < 10**-3):
			return w
			break
		else:
			loss = new_loss	
		for j in range(D):
			xij_factor = 0
			for i in range(X.shape[0]):
				xij_factor = X[i][j]**2 + xij_factor
				cj_factor = X[i][j] *(y[i] - np.dot(w.T,X[i]) + w[j]*X[i][j])
			aj = 2*xij_factor
			cj = 2*cj_factor
			if (cj > lamda):
				w[j] = 1/aj*(cj - lamda)
			elif (cj < -lamda):
				w[j] = 1/aj*(cj + lamda)
			elif (-lamda <= cj <= lamda):
				w[j] = 0
			
		new_loss = compute_loss(X,y,w)
		 		

#def coordinate_descent_vectorized(X,y,lamda,w):
	

def main():

	X = np.random.rand(150,75)
	global N,D
	N,D = X.shape

	theta_1 = np.empty(5)
	theta_1.fill(10)

	theta_2 = np.empty(5)
	theta_2.fill(-10)

	true_theta = np.hstack((theta_1,theta_2))
	np.random.shuffle(true_theta)

	theta_rest = np.zeros(65)
	true_theta = np.hstack((true_theta,theta_rest))

	epsilon = 0.1 * np.random.randn(150) + 0
	y = np.dot(X,true_theta) + epsilon
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=42)
	X_test,X_validation,y_test,y_validation = train_test_split(X_test, y_test, test_size=20, random_state=42)

	## 1.2 ridge regression experiments
	lambda_loss = np.zeros((11,2))
	lambda_theta = np.zeros((11,D))
	for i in range(-5,6):
	  Lambda = 10**i;
	  theta_opt = minimize(ridge(X_train,y_train,np.zeros(D),Lambda), np.ones(D))
	  lambda_loss[i] = Lambda, compute_loss(X_validation,y_validation,theta_opt.x) 
	  lambda_theta[i] = theta_opt.x

	least_loss_lambda = lambda_loss[np.argmin(lambda_loss,axis=0)[1],[0]]
	print 'least loss' , np.amin(lambda_loss,axis=0)[1]
	theta_learned = lambda_theta[np.argmin(lambda_loss,axis=0)[1]]

	#Report on how many components with true value 0 reported as non-zero and vice-versa

	plt.ylim(-1,1)
	plt.plot(theta_learned,'r',label = "ridge solution")
	plt.plot(true_theta,'g',label="true solution")
	legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    	legend.get_frame().set_facecolor('#00FFCC')
	plt.title('Ridge solution Vs True solution')
	plt.savefig('Ridge_solution.png')	
	plt.close()	
	
	# Comparing with sklearn Ridge

	clf = Ridge(alpha=2*N*least_loss_lambda)
	clf.fit(X_train, y_train)
	sklearn_ridge_loss = mean_squared_error(clf.predict(X_validation), y_validation) / 2
	print 'sklearn Ridge Regression loss is  ' , sklearn_ridge_loss
	sklearn_ridge_coef = clf.coef_

	# plot for comparison between Ridge regression and sklearn ridge solution 

	plt.ylim(-1,1)
	plt.plot(theta_learned,'r',label = "ridge solution")
	plt.plot(sklearn_ridge_coef,'g',label="sklearn solution")
	legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    	legend.get_frame().set_facecolor('#00FFCC')
	plt.title('Ridge solution Vs sklearn Ridge solution')
	plt.savefig('sklearn_vs_ridge.png')	
	plt.close()	


	#Report after choosing a small threshold of 10**-3 and making all coefficients smaller than the threshold to zero
	theta_scaled = theta_learned
	theta_scaled[theta_scaled < 10**-3] = 0
	plt.ylim(-1,1)
	plt.plot(theta_scaled,'r',label = "ridge solution scaled")
	plt.plot(true_theta,'g',label="true solution")
	legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    	legend.get_frame().set_facecolor('#00FFCC')
	plt.title('Ridge scaled solution Vs True solution')
	plt.savefig('Ridge_scaled_solution.png')	
	plt.close()	


	###########################################################################
	
	## 2.1 Coordinate descent Function for 10 diffenent lambdas ranging from 10 ** -5 to 10**6
	lasso_solutions = np.zeros((15,D))
	sq_loss_val_set = np.zeros(15)
	sq_loss_train = np.zeros(15)
	lamda_range = np.zeros(15)
	for i in range(-7,8):
		lamda = 10**i
		lamda_range[i] = lamda
		w_starting_point =  theta_learned#np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train) + lamda * np.identity(X_train.shape[1])),X_train.T),y_train)#np.zeros(D)
		lasso_solutions[i] = coordinate_descent(X_train,y_train,lamda,w_starting_point)
		sq_loss_val_set[i] = compute_loss(X_validation,y_validation,lasso_solutions[i])
		sq_loss_train[i] = compute_loss(X_train,y_train,lasso_solutions[i])
	
	best_lamda = lamda_range[np.argmin(sq_loss_val_set)]
	test_lasso_sol = coordinate_descent(X_test,y_test,best_lamda,w_starting_point)
	test_error = compute_loss(X_test,y_test,test_lasso_sol)
	print 'lamda which minimizes square  loss on validation set is ' , best_lamda
	print 'The corresponding test error is ' , test_error
	plt.plot(np.log(lamda_range),sq_loss_val_set,'g',label = "Validation error Vs lamda")
	plt.title('Validation square loss Vs lambda')
	plt.xlabel('lamda')
	plt.ylabel('square_loss')
	plt.savefig('sq_loss_vs_lamda.png')
	print sq_loss_val_set		 
	print sq_loss_train
	plt.close()
	####################################################################################################################################################

	## 2.2 Analyze the sparsity of solution
	print lasso_solutions[np.argmin(sq_loss_val_set)]
	plt.ylim(-1,1)
	plt.plot(lasso_solutions[np.argmin(sq_loss_val_set)],'r',label = "lasso solution")
	plt.plot(true_theta,'g',label="true solution")
	legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    	legend.get_frame().set_facecolor('#00FFCC')
	plt.title('sparsity in lasso solution Vs True solution')
	plt.savefig('sparsity.png')	
	plt.close()
	######################################################################################################################################################	

	## 2.3 homotopy Method Vs Basic Shooting Algorithm
	lamda_max = 2 * np.linalg.norm(np.dot(X_train.T,y_train))
	i = 0
	w_starting_point = np.zeros(D)
	lamda_len = 0
	lamda = lamda_max
	while(lamda >= 10 ** -5):
		lamda_len = lamda_len + 1
		lamda = lamda / 2
	w = np.zeros((lamda_len,D))
	
	#Run time for Basic Shooting Algorithm
	w_starting_point = np.zeros(D)
	i = 0
	lamda = lamda_max
	start_time = time.clock()
	shooting_time = 0
	while(lamda >= 10 ** -5):
		start_time = time.clock()
		w[i] = coordinate_descent(X_train,y_train,lamda,w_starting_point)
		end_time = time.clock()

		shooting_time = shooting_time + (end_time - start_time)

		lamda = float(lamda) / 10
		i = i + 1
	print 'Time taken by shooting algorithm  ', shooting_time
	#Run-time for homotopy method
	lamda = lamda_max
	homotopy_time = 0
	while(lamda >= 10 ** -5):
		start_time = time.clock()
		w[i] = coordinate_descent(X_train,y_train,lamda,w_starting_point)
		end_time = time.clock()
		
		homotopy_time = homotopy_time + (end_time - start_time)
		w_starting_point = w[i]
		lamda = float(lamda) / 10
		i = i + 1

	print 'Time taken by homotopy method is  ' , homotopy_time

	###########################################################################################################################################################

	#2.1.4 Vecotrized code for Shooting algorithm

	
	
if __name__ == "__main__":
    main()
