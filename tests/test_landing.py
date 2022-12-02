import numpy as np
import unittest

from scotty.landing import ParamSet, solve_min_fuel_and_time

vertical_descent = ParamSet(
    q=np.zeros(3),
    w=np.array([2.53e-5, 0, 6.62e-5]),
    g=3.71,
    min_thr=0.0 * 24000,
    max_thr=1.0 * 24000,
    ve=2000,
    mw=2000,
    md=300,
    r0=np.array([2400, 0, 0]),
    v0=np.array([-10, 0, 0])
)

large_divert = ParamSet(
    q=np.zeros(3),
    w=np.array([2.53e-5, 0, 6.62e-5]),
    g=3.71,
    min_thr=0.2 * 24000,
    max_thr=0.8 * 24000,
    ve=2000,
    mw=2000,
    md=300,
    r0=np.array([2400, 450, -330]),
    v0=np.array([-10, -40, 10])
)

infeasible_target = ParamSet(
    q=np.zeros(3),
    w=np.array([2.53e-5, 0, 6.62e-5]),
    g=3.71,
    min_thr=0.2 * 24000,
    max_thr=0.8 * 24000,
    ve=2000,
    mw=2000,
    md=1820,
    r0=np.array([2400, 450, -330]),
    v0=np.array([-10, -40, 10])
)


class MinFuelTestCase(unittest.TestCase):
    def test_vertical_descent(self):
        sol = solve_min_fuel_and_time(vertical_descent)
        np.testing.assert_almost_equal(sol.x.r[-1, :], np.zeros(3))
        np.testing.assert_almost_equal(sol.x.v[-1, :], np.zeros(3))
        np.testing.assert_almost_equal(sol.x.m[-1], 1846.18, decimal=1)

    def test_large_divert(self):
        sol = solve_min_fuel_and_time(large_divert)
        np.testing.assert_almost_equal(sol.x.r[-1, :], np.zeros(3))
        np.testing.assert_almost_equal(sol.x.v[-1, :], np.zeros(3))
        np.testing.assert_almost_equal(sol.x.m[-1], 1799.19, decimal=1)

    def test_infeasible_target(self):
        sol = solve_min_fuel_and_time(infeasible_target)
        np.testing.assert_almost_equal(sol.x.v[-1, :], np.zeros(3))
