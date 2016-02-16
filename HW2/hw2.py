import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from numpy.random import RandomState
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
	max_iter = 10000
	print 'lamda in coordinte descent function  ' , lamda	
	iter = 0
	converged = False
	cjval = np.zeros(D)
	XX2 = np.dot(X.T,X)*2
	Xy2 = np.dot(X.T,y)*2;
	while((not converged) and (iter < max_iter)):
		w_old = w
		for j in range(D):
			xij_factor = 0
			cj_factor = 0
			for i in range(X.shape[0]):
				xij_factor = X[i][j]**2 + xij_factor
				cj_factor = cj_factor + (X[i][j]*(y[i] - np.dot(w.T,X[i]) + w[j]*X[i][j]))

			aj = 2*xij_factor
			cj = 2*cj_factor
			
			aj_vec = XX2[j][j]
			cj_vec = 2 * np.sum(np.multiply(X[:,j],(y-np.dot(X,w.T)+w[j]*X[:,j]))) 
			
			if( ((aj - aj_vec) != 0) or ((cj - cj_vec) != 0)):
				print "iteration " , j
				print "---------------------"
				print aj, "    " ,aj_vec, " ", aj-aj_vec
				print cj, "    " , cj_vec, " ", cj - cj_vec


			if (cj > lamda):
				w[j] = 1/aj*(cj - lamda)
			elif (cj < -lamda):
				w[j] = 1/aj*(cj + lamda)
			else: 
				w[j] = 0
			iter = iter + 1
		
		converged = np.linalg.norm(np.absolute(w - w_old)) < 10 ** -3
		#converged = abs(compute_loss(X,y,w) - compute_loss(X,y,w_old)) < 10 ** -3
	return w	

def coordinate_descent_vectorized(X,y,lamda,w):
	max_iter = 10000
	print 'lamda in coordinte descent function  ' , lamda	
	iter = 0
	XX2 = np.dot(X.T,X)*2
	Xy2 = np.dot(X.T,y)*2;
	converged = False
	cjval = np.zeros(D)
	while((not converged) and (iter < max_iter)):
		w_old = w
		for j in range(D):
			aj = XX2[j][j]
			cj = Xy2[j] - np.sum(np.multiply(X[:,j],np.dot(X,w.T))) + XX2[j][j]*w[j]
			cjval[j] = cj
			if (cj > lamda):
				w[j] = 1/aj*(cj - lamda)
			elif (cj < -lamda):
				w[j] = 1/aj*(cj + lamda)
			else: 
				w[j] = 0
			iter = iter + 1
			
		#converged = np.linalg.norm(np.absolute(w - w_old)) < 10 ** -3
		converged = abs(compute_loss(X,y,w) - compute_loss(X,y,w_old)) < 10 ** -3
	print "vectorixed"  , cjval
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


	#np.random.seed(19910420)
	randState = RandomState(19910420)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=randState)
	X_test,X_validation,y_test,y_validation = train_test_split(X_test, y_test, test_size=20, random_state=randState)

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
		w_starting_point =  np.zeros(D)#theta_learned#np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train) + lamda * np.identity(X_train.shape[1])),X_train.T),y_train)#np.zeros(D)
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
	lamda = lamda_max
	
	#Run time for Basic Shooting Algorithm
	lamda = lamda_max
	shooting_time = 0
	while(lamda >= 10 ** -5):
		w = np.zeros(D)
		start_time = time.time()
		w = coordinate_descent(X_train,y_train,lamda,w)
		end_time = time.time()

		shooting_time = shooting_time + (end_time - start_time)

		lamda = float(lamda) / 2
	print 'Time taken by shooting algorithm  ', shooting_time
	#Run-time for homotopy method
	lamda = lamda_max
	homotopy_time = 0
	w = np.zeros(D)
	while(lamda >= 10 ** -5):
		start_time = time.time()
		w = coordinate_descent(X_train,y_train,lamda,w)
		end_time = time.time()
		
		homotopy_time = homotopy_time + (end_time - start_time)
		lamda = float(lamda) / 2
		i = i + 1

	print 'Time taken by homotopy method is  ' , homotopy_time

	###########################################################################################################################################################

	#2.1.4 Vecotrized code for Shooting algorithm
	coordinate_descent(X_train,y_train,0.01,np.zeros(D))
	coordinate_descent_vectorized(X_train,y_train,0.01,np.zeros(D))
	X= np.arange(8).reshape(4,2)
	y = np.ones(4)


	
if __name__ == "__main__":
    main()
