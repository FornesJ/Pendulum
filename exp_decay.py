from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt 


# class ExponentialDecay takes in const a 
class ExponentialDecay:
    def __init__(self, a):
        self.a = a

    # solve method using solve_ivp from scipy.integrate library, returning t and u as arrays
    def solve(self, u0, T, dt):
        sol = solve_ivp(self, [0, T], (u0,), t_eval=np.linspace(0,T,round(T/dt)), max_step=dt)
        t = sol.t
        u = sol.y[0]
        return t, u

    # returns an ODE with parameters t and u when called
    def __call__(self, t, u):
        return -self.a*u


if __name__ == "__main__":
    # some random parameters:
    a = 0.3
    T = 30
    dt = 1.0
    u0 = 40.0

    # calling class and solve
    decay_model = ExponentialDecay(a)
    t, u = decay_model.solve(u0, T, dt)

    plt.plot(t, u)
    plt.xlabel('time')
    plt.ylabel('decay')
    plt.show()
