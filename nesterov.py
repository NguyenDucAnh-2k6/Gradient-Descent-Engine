import numpy as np
import matplotlib.pyplot as plt
class GradientDescentv4:
    def __init__(self, func, grad, X, eps=1e-6):
        self.func=func
        self.grad=grad
        self.X=X
        self.eps=eps
    def compute_numerical_gradients(self, at_pos=None):
        if at_pos is None:
            at_pos=self.X
        flat=at_pos.reshape(-1)
        shape=at_pos.shape
        grad_flat=np.zeros_like(flat)
        for i in range(flat.size):
            Xp=flat.copy()
            Xn=flat.copy()
            Xp[i]+=self.eps
            Xn[i]-=self.eps
            f_Xp=self.func(Xp.reshape(shape))
            f_Xn=self.func(Xn.reshape(shape))
            grad_flat[i]=(f_Xp-f_Xn)/(2*self.eps)
        return grad_flat.reshape(shape)
    def nesterov_gd(self, eta=1e-2, gamma=0.9, max_iter=300):
        v=np.zeros_like(self.X)
        for i in range(1, max_iter+1):
            at_pos=self.X-gamma*v   #Future position
            grad_at_pos=self.compute_numerical_gradients(at_pos=at_pos)
            #Update velocity
            v=gamma*v+eta*grad_at_pos
            self.X-=v
            print(f"Iter {i}, loss = {self.func(self.X)}")
        return self.X
N=200
X=np.random.rand(N,1)
y=1+3.5*X+0.2*np.random.rand(N,1)
one=np.ones((X.shape[0],1))
Xbar=np.concatenate((one,X), axis=1)
def loss(w):
    residual=y-Xbar.dot(w)
    return (0.5/N)*np.sum(residual**2)
def grad(w):
    return np.zeros_like(w)
w_init=np.random.randn(2,1)
engine=GradientDescentv4(loss, grad, w_init)
w_opt=engine.nesterov_gd(eta=1.0)
w_0=w_opt[0][0]
w_1=w_opt[1][0]
x0=np.linspace(0,1,2,endpoint=True)
y0=w_0+w_1*x0
plt.plot(X.T,y.T, 'b.')
plt.axis([0,1,0,10])
plt.plot(x0, y0,'y', linewidth=2)
plt.show()