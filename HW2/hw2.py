import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ridge(X,y,theta,Lambda):
	def ridge_obj(theta):
	  return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*N) + Lambda*(np.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(X,y,theta):
	return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*N)

def coordinate_descent(X,y,lamda,w):
	k = 0	
	while(k <= 10):
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
		k = k +1
	return w
		 		

def main():

	X = np.random.rand(150,75)
	global N,D
	N,D = X.shape

	theta_1 = np.empty(5)
	theta_1.fill(10)

	theta_2 = np.empty(5)
	theta_2.fill(-10)

	theta = np.hstack((theta_1,theta_2))
	np.random.shuffle(theta)

	theta_rest = np.zeros(65)
	theta = np.hstack((theta,theta_rest))

	epsilon = 0.1 * np.random.randn(150) + 0
	y = np.dot(X,theta) + epsilon
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=42)
	X_test,X_validation,y_test,y_validation = train_test_split(X_test, y_test, test_size=50, random_state=42)

	## 1.2 ridge regression experiments
	lambda_loss = np.zeros((11,2))
	lambda_theta = np.zeros((11,D))
	for i in range(-5,6):
	  Lambda = 10**i;
	  theta_opt = minimize(ridge(X_train,y_train,theta,Lambda), theta)
	  lambda_loss[i] = Lambda, compute_loss(X_validation,y_validation,theta_opt.x) 
	  lambda_theta[i] = theta_opt.x

	least_loss_lambda = lambda_loss[np.argmin(lambda_loss,axis=0)[1],[0]]
	print 'least loss' , np.amin(lambda_loss,axis=0)[1]
	theta_learned = lambda_theta[np.argmin(lambda_loss,axis=0)[1]]
	
	###########################################################################
	
	## 2.1 Coordinate descent Function for 10 diffenent lambdas ranging from 10 ** -5 to 10**6
	lasso_solutions = np.zeros((11,D))
	sq_loss_val_set = np.zeros(11)
	sq_loss_train = np.zeros(11)
	lamda_range = np.zeros(11)
	for i in range(-5,6):
		lamda = 10**i
		lamda_range[i] = lamda
		w_starting_point = np.zeros(D) #np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train) + lamda * np.identity(X_train.shape[1])),X.T),y_train)
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
	##############################################################################

	## 2.2 Analyze the sparsity of solution
	print lasso_solutions[np.argmin(sq_loss_val_set)]
	plt.ylim(-1,1)
	plt.plot(lasso_solutions[np.argmin(sq_loss_val_set)],'r',label = "lasso solution")
	plt.plot(theta,'g',label="true solution")
	legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
    	legend.get_frame().set_facecolor('#00FFCC')
	plt.title('sparsity in lasso solution Vs True solution')
	plt.savefig('sparsity.png')	
	
	
if __name__ == "__main__":
    main()
