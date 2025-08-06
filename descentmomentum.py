import numpy as np
import matplotlib.pyplot as plt
class GradientDescentv3:
    def __init__(self, func, grad, X, eps=1e-6):
        self.func = func
        self.grad = grad    # not used by momentum descent, but kept for API consistency
        self.X = X
        self.eps = eps

    def compute_numerical_gradient(self):
        X_flat = self.X.reshape(-1)
        shape_X = self.X.shape
        grad_flat = np.zeros_like(X_flat)
        for i in range(X_flat.shape[0]):
            Xp = X_flat.copy(); Xn = X_flat.copy()
            Xp[i] += self.eps
            Xn[i] -= self.eps
            grad_flat[i] = (self.func(Xp.reshape(shape_X))
                            - self.func(Xn.reshape(shape_X))) / (2*self.eps)
        return grad_flat.reshape(shape_X)

    def gradient_descent_momentum(self, lr=1e-2, gamma=0.9, max_iter=100):
        v = np.zeros_like(self.X)
        for i in range(1, max_iter+1):
            grad = self.compute_numerical_gradient()
            # velocity update
            v = gamma * v + grad
            # weight update
            self.X -= lr * v
            loss = self.func(self.X)
            print(f"Iter {i:3d}: Loss = {loss:.6f}")
        return self.X      
# -------------------------------------------------------
# Example: Use on least‐squares L(w) = (1/2N)||y - Xw||^2
# -------------------------------------------------------
N=1000
X=np.random.rand(N,1)
y=2+4*X+0.2*np.random.rand(N,1)
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
checker = GradientDescentv3(func_L, dummy_grad, w_init)
w_opt = checker.gradient_descent_momentum(lr=1.0, max_iter=300)
w_0 = w_opt[0][0]
w_1 = w_opt[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()