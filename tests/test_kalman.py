#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations
import time

import numpy as np
import minnow as mn
import jax

def test_kalman_runs_dm():
    # Test if the kalman filter runs
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/12p5_nodmx/B1855+09_12p5y_no_dmx.feather")

    kf_obj = mn.KalmanMultiBand(psr,
                   mn.make_residual_tracking_transition_matrix_dm(psr),
                   mn.make_design_matrix_dm(psr, 186.4940816702),
                   mn.make_covariance_matrices(psr, mn.covariance_matrix_dm),
                   mn.make_measurement_covar(psr,
                                             noisedict=psr.noisedict,
                                             tnequad=True))
    filter = kf_obj.makefilter()
    pardict = {}
    pardict['B1855+09_P0'] = np.ones(24) * 1e-20
    pardict['B1855+09_process_cov_sigma_dphi'] = 0
    pardict['B1855+09_process_cov_sigma_dm_dphi'] = 0
    pardict['B1855+09_process_cov_sigma_df'] = 1e-14
    pardict['B1855+09_process_cov_sigma_dfdot'] = 0
    pardict['B1855+09_x0'] = np.zeros(24)

    jfilt = jax.jit(filter)
    # run once to compile
    jfilt(pardict)
    start = time.time()
    for ii in range(100):
        out = jfilt(pardict)
    print('Time taken for 100 iterations:', time.time() - start)
    assert sorted(filter.params) == sorted(['B1855+09_P0', 'B1855+09_process_cov_sigma_df', 'B1855+09_process_cov_sigma_dfdot', 'B1855+09_process_cov_sigma_dphi', 'B1855+09_x0','B1855+09_process_cov_sigma_dm_dphi'])

def test_kalman_runs():
    # Test if the kalman filter runs
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/12p5_nodmx/B1855+09_12p5y_no_dmx.feather")

    kf_obj = mn.KalmanMultiBand(psr,
                   mn.make_residual_tracking_transition_matrix(psr),
                   mn.make_design_matrix(psr, 186.4940816702),
                   mn.make_covariance_matrices(psr, mn.covariance_matrix),
                   mn.make_measurement_covar(psr,
                                             noisedict=psr.noisedict,
                                             tnequad=True))
    filter = kf_obj.makefilter()
    pardict = {}
    pardict['B1855+09_P0'] = np.ones(3) * 1e-20
    pardict['B1855+09_process_cov_sigma_dphi'] = 0
    pardict['B1855+09_process_cov_sigma_df'] = 1e-14
    pardict['B1855+09_process_cov_sigma_dfdot'] = 0
    pardict['B1855+09_x0'] = np.zeros(3)

    jfilt = jax.jit(filter)
    # run once to compile
    jfilt(pardict)
    start = time.time()
    for ii in range(100):
        out = jfilt(pardict)
    print('Time taken for 100 iterations:', time.time() - start)
    assert sorted(filter.params) == sorted(['B1855+09_P0', 'B1855+09_process_cov_sigma_df', 'B1855+09_process_cov_sigma_dfdot', 'B1855+09_process_cov_sigma_dphi', 'B1855+09_x0'])

def test_sample_kalman():
    import numpyro
    import numpyro.distributions as dist
    from numpyro import infer
    from minnow.settings import jnp

    psr = mn.MultiBandPulsar.read_feather_pre_process("data/12p5_nodmx/B1855+09_12p5y_no_dmx.feather")
    rot_freq =  186.4940816702
    kf_obj = mn.KalmanMultiBand(psr,
                    mn.make_residual_tracking_transition_matrix_dm(psr),
                    mn.make_design_matrix_dm(psr, rot_freq),
                    mn.make_covariance_matrices_mb(psr, mn.covariance_matrix_mb_dm),
                    mn.make_measurement_covar(psr,
                                              noisedict=psr.noisedict,
                                              tnequad=True))
    filter = kf_obj.makefilter()
    jfilt = jax.jit(filter)

    def model_all(burn=0, rng_key=None):
        pardict = {}

        x0 = []
        x0.append(numpyro.sample('phi0', dist.Uniform(-1e-4, 1e-4), rng_key=rng_key).squeeze())
        x0.extend([0, 0])
        x0.append(numpyro.sample('dm0', dist.Uniform(-1e-1, 1e-1),rng_key=rng_key).squeeze())
        x0.extend([0]*20)
        P0 = []
        P0.append(10**numpyro.sample('P0_phi', dist.Uniform(-10, 4), rng_key=rng_key))
        P0.append(10**numpyro.sample('P0_f', dist.Uniform(-40, -15), rng_key=rng_key))
        P0.append(10**numpyro.sample('P0_fdot', dist.Uniform(-60, -40), rng_key=rng_key))
        P0.append(10**numpyro.sample('P0_DM', dist.Uniform(-10, -1), rng_key=rng_key))
        svd_val = 10**numpyro.sample('P0_svd', dist.Uniform(-10, -6).expand([20]), rng_key=rng_key)
        P0.extend(svd_val)
        pardict[f'{psr.name}_P0'] = jnp.array(P0).squeeze()
        pardict[f'{psr.name}_x0'] = jnp.array(x0)
        pardict[f'{psr.name}_process_cov_sigma_dphi'] = 10**numpyro.sample('\sigma_{phi}', dist.Uniform(-30, 0), rng_key=rng_key)
        pardict[f'{psr.name}_process_cov_sigma_dm_dphi'] = 10**numpyro.sample('\sigma_{DM}', dist.Uniform(-10, 1), rng_key=rng_key)
        pardict[f'{psr.name}_process_cov_sigma_df'] = 10**numpyro.sample('\sigma_{f}', dist.Uniform(-30, 0), rng_key=rng_key)
        pardict[f'{psr.name}_process_cov_sigma_dfdot'] = 10**numpyro.sample('\sigma_{fdot}', dist.Uniform(-40, -20), rng_key=rng_key)

        ll, xr, Pr, xp, Pp = jfilt(pardict)
        # print(ll)
        numpyro.deterministic('state_vals', xr)
        numpyro.deterministic('stat_covars', Pr)
        numpyro.deterministic('state_preds', xp)
        numpyro.deterministic('loglikes', jnp.sum(jnp.array(ll[burn:])))
        numpyro.factor('ll', jnp.sum(jnp.array(ll[burn:])))

    kernel = infer.NUTS(model_all, max_tree_depth=5, target_accept_prob=0.99)
    sampler = infer.MCMC(kernel, num_warmup=20, num_samples=20)
    sampler.run(jax.random.PRNGKey(43))