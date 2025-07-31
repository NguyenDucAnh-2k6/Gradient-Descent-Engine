import numpy as np
import matplotlib.pyplot as plt
class GradientDescentv2:
    def __init__(self, func, grad, X, eps=1e-6):
        self.func = func
        self.grad = grad
        self.X = X
        self.eps = eps

    def compute_numerical_gradient(self):
        X_flat = self.X.reshape(-1)
        shape_X = self.X.shape
        grad_flat = np.zeros_like(X_flat)
        for i in range(X_flat.shape[0]):
            Xp_flat = X_flat.copy()
            Xn_flat = X_flat.copy()
            Xp_flat[i] += self.eps
            Xn_flat[i] -= self.eps
            Xp = Xp_flat.reshape(shape_X)
            Xn = Xn_flat.reshape(shape_X)
            grad_flat[i] = (self.func(Xp) - self.func(Xn)) / (2 * self.eps)
        return grad_flat.reshape(shape_X)

    def gradient_descent(self, lr=1.0, max_iter=200, thresh=1e-6):
        for i in range(max_iter):
            grad = self.compute_numerical_gradient()
            self.X -= lr * grad
            loss = self.func(self.X)
            print(f"Iter {i+1}: Loss = {loss:.6f}")
            if np.linalg.norm(grad)<thresh:
                print('Final sol: ',self.X, 'after ', i, 'iterations.')
                break
        return self.X

# Example: Least squares loss
N=1000
X = np.random.rand(N, 1)
y = 1 + (4.5) * X + .2*np.random.randn(N, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

def func_L(w):
    # Compute loss: (1/2N) * ||y - Xw||^2
    residual = y - Xbar.dot(w)
    return (0.5 / N) * np.sum(residual**2)

# Use dummy grad since we’ll use numerical gradient anyway
def dummy_grad(w):
    return np.zeros_like(w)

# Initialize weight
w_init = np.random.randn(2, 1)

# Run
checker = GradientDescentv2(func_L, dummy_grad, w_init)
w_opt = checker.gradient_descent(lr=1.0, max_iter=200, thresh=1e-6)
w_0 = w_opt[0][0]
w_1 = w_opt[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()
