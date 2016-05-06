import numpy as np
from scipy.optimize import minimize

#X = numpy.loadtxt("X.txt")
#y = numpy.loadtxt("y.txt")

X = np.random.rand(150,75)
theta = np.empty(10)
theta.fill(10)

theta_1 = np.zeros(65)
theta = np.hstack((theta,theta_1))

epsilon = 0.1 * np.random.randn(75) + 0



(N,D) = X.shape

w = numpy.random.rand(D,1)

def ridge(Lambda):
  def ridge_obj(theta):
    return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
  return ridge_obj

def compute_loss(Lambda, theta):
  return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N)

for i in range(-5,6):
  Lambda = 10**i;
  w_opt = minimize(ridge(Lambda), w)
  print Lambda, compute_loss(Lambda, w_opt.x)

