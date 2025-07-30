import numpy as np
import sympy as sp

class SymbolicGradientDescent:
    def __init__(self,
                 func_str: str,
                 eta: float,
                 thresh: float,
                 max_iters: int,
                 var_name: str = 'x'):
        """
        func_str  : cost function as a string, e.g. "x**2 + 5*sin(x)"
        eta       : learning rate
        thresh    : stop when |grad| < thresh
        max_iters : maximum GD iterations
        var_name  : name of your independent variable in func_str
        """
        # 1) create a symbolic variable
        self.x_sym = sp.symbols(var_name)

        # 2) parse your function
        self.f_expr = sp.sympify(func_str)

        # 3) compute its derivative
        self.df_expr = sp.diff(self.f_expr, self.x_sym)

        # 4) make fast numerical functions via lambdify
        #    'numpy' backend → exactly match f(x) and grad(x) to arrays/scalars
        self.f  = sp.lambdify(self.x_sym, self.f_expr,  'numpy')
        self.df = sp.lambdify(self.x_sym, self.df_expr, 'numpy')

        # store GD hyper‑parameters
        self.eta       = eta
        self.thresh    = thresh
        self.max_iters = max_iters

    def run(self, x0: float) -> tuple[list[float], int]:
        """
        Perform GD starting at x0.
        Returns: (path_of_x, iters_used)
        """
        path = [x0]
        for i in range(1, self.max_iters+1):
            grad_val = self.df(path[-1])
            if abs(grad_val) < self.thresh:
                return path, i-1
            path.append(path[-1] - self.eta * grad_val)
        return path, self.max_iters

    def cost(self, x: float) -> float:
        """Evaluate cost at x."""
        return float(self.f(x))


if __name__=='__main__':
  print("Enter function: ", end=' ')
  func=str(input())
  print("Enter learning rate", end=' ')
  eta=float(input())
  print("Enter max iterations", end=' ')
  max_iters=int(input())
  print("Enter threshold", end=' ')
  thresh=float(input())
  print("Starting point (positive): ", end=' ')
  x0=float(input())
  solver=SymbolicGradientDescent(func_str=func, eta=eta, thresh=thresh, max_iters=max_iters, var_name='x')
  x1_path, iters1=solver.run(x0)
  x2_path, iters2=solver.run(-x0)
  #Final sols
  x1=x1_path[-1]
  x2=x2_path[-1]
  print(f"Solution for x1: {x1}, cost={solver.cost(x1):.6f}, after {iters1} iterations.")
  print(f"Solution for x2: {x2}, cost={solver.cost(x2):.6f}, after {iters2} iterations.")
