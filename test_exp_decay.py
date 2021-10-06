import pytest
from scipy.integrate import solve_ivp
from exp_decay import ExponentialDecay
import numpy as np


# defining paramiters u, t, const a and answer du
# function testing ExponentialDecay
@pytest.mark.parametrize("a, u, t, du", [(0.4, 3.2, 0, -1.28)])
def test_ExponentialDecay(a, u, t, du):
    if a < 0:
        raise ValueError
    else:
        test = ExponentialDecay(a)
        tol = 1e-6
        assert test(t, u) - du < tol

# testing function fun and exponentialDecay is equal
@pytest.mark.parametrize("a, T, dt, u0", [(0.4, 30, 1.0, 1.28)])
def test_solve(a, T, dt, u0):
    test = ExponentialDecay(a)
    fun = lambda t, u: u0*np.exp(-a*t)
    t, u = test.solve(u0, T, dt)
    answer = solve_ivp(fun, [0, T], (u0,), t_eval=np.linspace(0,T,round(T/dt)), max_step=dt)
    assert u.all() == answer.y[0].all()

    




    

