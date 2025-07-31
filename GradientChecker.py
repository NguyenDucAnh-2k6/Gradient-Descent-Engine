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
#Example 1: grad(trace(A*X)) == A.T  
m,n=10,20
A=np.random.rand(m,n)
X=np.random.rand(n,m)
def func1(X):
    return np.trace(A.dot(X))
def grad1(X):
    return A.T
checker1=GradientChecker(func1, grad1, X)
checker1.check()
#Example 2: grad(trace(x.T*A*x)==(A+A.T)*x)
A=np.random.rand(m,m)
x=np.random.rand(m,1)
def func2(x):
    return x.T.dot(A).dot(x).item()
def grad2(x):
    return (A+A.T).dot(x)
checker2=GradientChecker(func2, grad2, x)
checker2.check()

 



