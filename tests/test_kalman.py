#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

from scipy.stats import kstest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import minnow as mn

def test_kalman_runs():
    # Test if the kalman filter runs
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")

    kf_obj = mn.Kalman_varQ(psr,
                   mn.make_residual_tracking_transition_matrix(psr),
                   mn.make_design_matrix(psr, 186.4940816702),
                   mn.make_covariance_matrices_mb(psr, mn.covariance_matrix_mb),
                   mn.make_measurement_covar(psr, noisedict=psr.noisedict))
    filter = kf_obj.makefilter()
    pardict = {}
    pardict['B1855+09_P0'] = np.ones(3) * 1e-20
    pardict['B1855+09_process_cov_sigma_dphi'] = 0
    pardict['B1855+09_process_cov_sigma_df'] = 1e-14
    pardict['B1855+09_process_cov_sigma_dfdot'] = 0
    pardict['B1855+09_x0'] = np.zeros(3)

    out = filter(pardict)

    assert filter.params == ['B1855+09_P0', 'B1855+09_process_cov_sigma_df', 'B1855+09_process_cov_sigma_dfdot', 'B1855+09_process_cov_sigma_dphi', 'B1855+09_x0']