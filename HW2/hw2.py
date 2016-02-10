import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import train_test_split



def ridge(Lambda):
	def ridge_obj(theta):
	  return ((np.linalg.norm(np.dot(X_train,theta) - y_train))**2)/(2*N) + Lambda*(np.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(theta):
	return ((np.linalg.norm(np.dot(X_validation,theta) - y_validation))**2)/(2*N)

def coordinate_descent(X,y,lamda,w):
	
	while(k <= 100):
		for j in range(D):
			xij_factor = 0
			for i in range(X.shape[0]):
				xij_factor = X[i][j]**2 + xij_factor
				cj_factor = X[i][j] *(y[i] - np.dot(w.T,X[i]) + w[j]X[i][j])
			aj = 2*xij_factor
			cj = 2*cj_factor
			w[j] = 
		 


def main():

	 X = np.random.rand(150,75)
	(N,D) = X.shape

	theta_1 = np.empty(5)
	theta_1.fill(10)

	theta_2 = np.empty(5)
	theta_2.fill(-10)

	theta = np.hstack((theta_1,theta_2))
	np.random.shuffle(theta)

	theta_rest = np.zeros(65)
	theta = np.hstack((theta,theta_rest))

	epsilon = 0.1 * np.random.randn(150) + 0

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=70, random_state=42)
	X_test,X_validation,y_test,y_validation = train_test_split(X_test, y_test, test_size=50, random_state=42)

	## 1.2 ridge regression experiments
	for i in range(-5,6):
	  Lambda = 10**i;
	  theta_opt = minimize(ridge(Lambda), theta)
	  lambda_loss[i] = Lambda, compute_loss(theta_opt.x) 
	  lambda_theta[i] = theta_opt.x

	least_loss_lambda = lambda_loss[np.argmin(lambda_loss,axis=0)[1],[0]]
	print 'least loss' , np.amin(lambda_loss,axis=0)[1]
	theta_learned = lambda_theta[np.argmin(lambda_loss,axis=0)[1]]
	
	###########################################################################
	
	## 2.1 Coordinate descent Function for 10 diffenent lambdas ranging from 10 ** -5 to 10**6
	lasso_solutions = np.zeros(11,D)
	for i in range(-5,6):
		lamda = 10**i
		w_starting_point =np.dot(np.dot(np.linalg.inverse(np.dot(X_train.T,X_train) + lamda * np.identity(X_train.shape[1])),X.T),y_train)
		lasso_solutions[i] = coordinate_descent(X_train,y_train,lamda,w_starting_point)
		 

	
	
if __name__ == "__main__":
    main()
