import numpy as np
import matplotlib.pyplot as plt 

class log2dGauss:
    def __init__(self, mu, L):
        self.mu = mu
        self.L = L
    
    def eval(self, x):
        inv_L = np.linalg.inv(L)
        aux = inv_L @ (x - mu)
        out = -0.5*np.shape(mu)[0]*np.log(2*np.pi) - np.sum(np.log(np.diag(L))) - 0.5*aux.T @ aux
        grad = -inv_L.T @ aux
        return (out, grad)

mu = np.asarray([0, 0])
sigma = np.asarray([[1, 0.95], [0.95, 1]])
L = np.linalg.cholesky(sigma)
dist = log2dGauss(mu, L)
