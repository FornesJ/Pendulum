import pytest
from scipy.integrate import solve_ivp
from pendulum import Pendulum
import numpy as np
from math import isclose

@pytest.mark.parametrize("L, theta, omega, output", [(2.7, np.pi/6, 0.15, -1.816666667)])
def test_pendulum(L, theta, omega, output):
    test = Pendulum(L)
    t = 1
    y = theta,omega
    try:
        assert test(t, y)[1] == output
        assert test(t, y)[0] == omega
    except AssertionError:
        isclose(test(t, y)[1], output, rel_tol=1e-6, abs_tol=0)
        assert test(t, y)[0] == omega

@pytest.mark.parametrize("L, theta, omega, output", [(2.7, 0, 0, 0)])
def test_pendulum_zero(L, theta, omega, output):
    test = Pendulum(L)
    t = 1
    y = theta,omega
    assert test(t, y) == (output,omega)

@pytest.mark.parametrize("L, y0, T, dt", [(2.7, (3,3), 10, 1.0)])
def test_property_solve(L, y0, T, dt):
    test = Pendulum(L)
    try:
        print(test.theta)
    except:
        ValueError

@pytest.mark.parametrize("L, y0, T, dt", [(2.7, (0,0), 10, 0.1)])
def test_solve_zero(L, y0, T, dt):
    test = Pendulum(L)
    test.solve(y0, T, dt)
    for i in range(len(test.theta)):
        assert test.theta[i] == 0
    for i in range(len(test.omega)):
        assert test.omega[i] == 0
    for i in range(len(test.t)):
        try:
            assert test.t[i] == i*dt
        except AssertionError:
            isclose(test.t[i], i*dt, rel_tol=1e-9, abs_tol=0)





 