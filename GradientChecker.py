import numpy as np
import copy
class GradientChecker:
    def __init__(self, func, grad, X, eps=1e-6):
        self.func=func  #Function to compute value
        self.grad=grad  #Function to compute analytic grad
        self.X=X
        self.eps=eps
    def compute_numerical_gradient(self):
        X_flat=self.X.reshape(-1) #for looping ease when +- eps
        shape_X=self.X.shape
        grad_flat=np.zeros_like(X_flat)
        for i in range(X_flat.shape[0]):
            Xp_flat=X_flat.copy()
            Xn_flat=X_flat.copy()
            Xp_flat[i]+=self.eps
            Xn_flat[i]-=self.eps
            Xp=Xp_flat.reshape(shape_X)
            Xn=Xn_flat.reshape(shape_X)
            grad_flat[i]=(self.func(Xp)-self.func(Xn))/(2*self.eps)
        return grad_flat.reshape(shape_X)
    def check(self):
        num_grad=self.compute_numerical_gradient()
        analytical_grad=self.grad(self.X)
        diff=np.linalg.norm(num_grad-analytical_grad)
        print('Computational difference: ', diff)
#Example 1: grad(trace(X)) == I  
def func7(X):
  return np.trace(X)
def grad7(X):
  return np.eye(X.shape[0])
X=np.random.rand(m,m)
checker7=GradientChecker(func7, grad7, X, eps=1e-6)
checker7.check()
#Example 2: grad(trace(A.T*X*B)==A*B.T)
def func2(X):
  return np.trace(A.T.dot(X).dot(B))
def grad2(X):
  return A.dot(B.T)
A=np.random.rand(m,n)
X=np.random.rand(m,n)
B=np.random.rand(n,n)
checker2=GradientChecker(func2, grad2, X, eps=1e-6)
checker2.check()

 



