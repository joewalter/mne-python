# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Daniel Strohmeier <daniel.strohmeier@gmail.com>
#
# License: Simplified BSD

import numpy as np
import warnings
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose, assert_array_less, assert_equal)

from mne.inverse_sparse.mxne_optim import (mixed_norm_solver,
                                           tf_mixed_norm_solver,
                                           iterative_mixed_norm_solver,
                                           norm_epsilon_inf, norm_epsilon,
                                           _Phi, _PhiT, dgap_l21l1)
from mne.time_frequency.stft import stft_norm2

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _generate_tf_data():
    n, p, t = 30, 40, 64
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    active_set = [0, 4]
    times = np.linspace(0, 2 * np.pi, t)
    X[0] = np.sin(times)
    X[4] = -2 * np.sin(4 * times)
    X[4, times <= np.pi / 2] = 0
    X[4, times >= np.pi] = 0
    M = np.dot(G, X)
    M += 1 * rng.randn(*M.shape)
    return M, G, active_set


def test_l21_mxne():
    """Test convergence of MxNE solver"""
    n, p, t, alpha = 30, 40, 20, 1.
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    args = (M, G, alpha, 1000, 1e-8)
    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=None,
        debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _, gap_cd = mixed_norm_solver(
        *args, active_set_size=None,
        debias=True, solver='cd', return_gap=True)
    assert_array_less(gap_cd, 1e-8)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, E, gap_bcd = mixed_norm_solver(
        M, G, alpha, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='bcd', return_gap=True)
    assert_array_less(gap_bcd, 9.6e-9)
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_prox, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_prox, X_hat_bcd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)

    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=2, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])

    # suppress a coordinate-descent warning here
    with warnings.catch_warnings(record=True):
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)
    assert_allclose(X_hat_bcd, X_hat_cd, rtol=1e-2)

    X_hat_bcd, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=5, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    X_hat_prox, active_set, _ = mixed_norm_solver(
        *args, active_set_size=2, debias=True, n_orient=5, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with warnings.catch_warnings(record=True):  # coordinate-ascent warning
        X_hat_cd, active_set, _ = mixed_norm_solver(
            *args, active_set_size=2, debias=True, n_orient=5, solver='cd')

    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd)
    assert_allclose(X_hat_bcd, X_hat_prox, rtol=1e-2)


def test_tf_mxne():
    """Test convergence of TF-MxNE solver"""
    alpha_space = 10.
    alpha_time = 5.

    M, G, active_set = _generate_tf_data()

    X_hat_tf, active_set_hat_tf, E, gap_tfmxne = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
        n_orient=1, tstep=4, wsize=32, return_gap=True)
    assert_array_less(gap_tfmxne, 1e-8)
    assert_array_equal(np.where(active_set_hat_tf)[0], active_set)

    alpha_space = 1e8
    alpha_time = 1e8
    X_hat_tf, active_set_hat_tf, E, gap_tfmxne = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
        n_orient=1, tstep=4, wsize=32, return_gap=True)
    assert_array_less(gap_tfmxne, 1e-8)
    assert_equal(X_hat_tf.shape[0], 0)
    assert_equal(np.any(active_set_hat_tf), False)


def test_norm_epsilon():
    """Test computation of espilon norm on TF coefficients."""
    n_steps = 5
    n_freqs = 4
    Y = np.zeros(n_steps * n_freqs)
    l1_ratio = 0.5
    assert_allclose(norm_epsilon(Y, l1_ratio, n_steps), 0.)

    Y[0] = 2.
    assert_allclose(norm_epsilon(Y, l1_ratio, n_steps), np.max(Y))

    l1_ratio = 1.
    assert_allclose(norm_epsilon(Y, l1_ratio, n_steps), np.max(Y))
    # dummy value without random:
    Y = np.arange(n_steps * n_freqs).reshape(-1, )
    l1_ratio = 0.
    assert_allclose(norm_epsilon(Y, l1_ratio, n_steps) ** 2,
                    stft_norm2(Y.reshape(-1, n_freqs, n_steps)))


def test_dgapl21l1():
    """Test duality gap for L21 + L1 regularization."""
    n_orient = 2
    M, G, active_set = _generate_tf_data()
    n_times = M.shape[1]
    n_sources = G.shape[1]
    tstep, wsize = 4, 32
    n_steps = int(np.ceil(n_times / float(tstep)))
    n_freqs = wsize // 2 + 1
    n_coefs = n_steps * n_freqs
    phi = _Phi(wsize, tstep, n_coefs)
    phiT = _PhiT(tstep, n_freqs, n_steps, n_times)

    for l1_ratio in [0.05, 0.1]:
        alpha_max = norm_epsilon_inf(G, M, phi, l1_ratio, n_orient)
        alpha_space = (1. - l1_ratio) * alpha_max
        alpha_time = l1_ratio * alpha_max

        Z = np.zeros([n_sources, n_coefs])
        shape = (-1, n_steps, n_freqs)
        # for alpha = alpha_max, Z = 0 is the solution so the dgap is 0
        gap = dgap_l21l1(M, G, Z, np.ones(n_sources, dtype=bool),
                         alpha_space, alpha_time, phi, phiT, shape, n_orient,
                         -np.inf)[0]

        assert_allclose(0., gap)
        # check that solution for alpha smaller than alpha_max is non 0:
        X_hat_tf, active_set_hat_tf, E, gap = tf_mixed_norm_solver(
            M, G, alpha_space / 1.01, alpha_time / 1.01, maxit=200, tol=1e-8,
            verbose=True, debias=False, n_orient=n_orient, tstep=tstep,
            wsize=wsize, return_gap=True)
        # allow possible small numerical errors (negative gap)
        assert_array_less(-1e-10, gap)
        assert_array_less(gap, 1e-8)
        assert_array_less(1, len(active_set_hat_tf))

        X_hat_tf, active_set_hat_tf, E, gap = tf_mixed_norm_solver(
            M, G, alpha_space / 5., alpha_time / 5., maxit=200, tol=1e-8,
            verbose=True, debias=False, n_orient=n_orient, tstep=tstep,
            wsize=wsize, return_gap=True)
        assert_array_less(-1e-10, gap)
        assert_array_less(gap, 1e-8)
        assert_array_less(1, len(active_set_hat_tf))


def test_tf_mxne_vs_mxne():
    """Test equivalence of TF-MxNE (with alpha_time=0) and MxNE"""
    alpha_space = 60.
    alpha_time = 0.

    M, G, active_set = _generate_tf_data()

    X_hat_tf, active_set_hat_tf, E = tf_mixed_norm_solver(
        M, G, alpha_space, alpha_time, maxit=200, tol=1e-8, verbose=True,
        debias=False, n_orient=1, tstep=4, wsize=32)

    # Also run L21 and check that we get the same
    X_hat_l21, _, _ = mixed_norm_solver(
        M, G, alpha_space, maxit=200, tol=1e-8, verbose=False, n_orient=1,
        active_set_size=None, debias=False)

    assert_allclose(X_hat_tf, X_hat_l21, rtol=1e-1)


def test_iterative_reweighted_mxne():
    """Test convergence of irMxNE solver"""
    n, p, t, alpha = 30, 40, 20, 1
    rng = np.random.RandomState(0)
    G = rng.randn(n, p)
    G /= np.std(G, axis=0)[None, :]
    X = np.zeros((p, t))
    X[0] = 3
    X[4] = -2
    M = np.dot(G, X)

    X_hat_l21, _, _ = mixed_norm_solver(
        M, G, alpha, maxit=1000, tol=1e-8, verbose=False, n_orient=1,
        active_set_size=None, debias=False, solver='bcd')
    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
        debias=False, solver='bcd')
    X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 1, maxit=1000, tol=1e-8, active_set_size=None,
        debias=False, solver='prox')
    assert_allclose(X_hat_bcd, X_hat_l21, rtol=1e-3)
    assert_allclose(X_hat_prox, X_hat_l21, rtol=1e-3)

    X_hat_prox, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='prox')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
        debias=True, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=None,
        debias=True, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 4])
    assert_array_almost_equal(X_hat_prox, X_hat_cd, 5)
    assert_array_almost_equal(X_hat_bcd, X_hat_cd, 5)

    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
        debias=True, n_orient=2, solver='bcd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    # suppress a coordinate-descent warning here
    with warnings.catch_warnings(record=True):
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=2, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 4, 5])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)

    X_hat_bcd, active_set, _ = iterative_mixed_norm_solver(
        M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2, debias=True,
        n_orient=5)
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    with warnings.catch_warnings(record=True):  # coordinate-ascent warning
        X_hat_cd, active_set, _ = iterative_mixed_norm_solver(
            M, G, alpha, 5, maxit=1000, tol=1e-8, active_set_size=2,
            debias=True, n_orient=5, solver='cd')
    assert_array_equal(np.where(active_set)[0], [0, 1, 2, 3, 4])
    assert_array_equal(X_hat_bcd, X_hat_cd, 5)
