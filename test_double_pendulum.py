import pytest
from scipy.integrate import solve_ivp
from double_pendulum import DoublePendulum
import numpy as np
from math import isclose

@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,            0),
        (  0, 0.5,  3.386187037), 
        (0.5,   0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ]
)
def test_domega1_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = dp(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)
    
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,          0.0),
        (  0, 0.5, -7.704787325),
        (0.5,   0,  6.768494455),
        (0.5, 0.5,          0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = dp(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)