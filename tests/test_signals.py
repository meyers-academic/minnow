import discovery as ds
import minnow as mn
import numpy as np

def test_make_residual_tracking_transition_matrix_dm():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    transition_matrices = mn.make_residual_tracking_transition_matrix_dm(psr)
    assert transition_matrices.shape == (360, 4, 4)

    # We should start with small step.
    assert np.allclose(transition_matrices[0, :, :], [[1, 1, 0.5, 0],
                                                      [0, 1, 1, 0],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]])

def test_make_residual_tracking_transition_matrix():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    transition_matrices = mn.make_residual_tracking_transition_matrix(psr)
    assert transition_matrices.shape == (360, 3, 3)

    # We should start with small step.
    assert np.allclose(transition_matrices[0, :, :], [[1, 1, 0.5],
                                                      [0, 1, 1],
                                                      [0, 0, 1]])

def test_make_design_matrix_dm():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    frot_dummy = np.random.rand() * 1000
    design_matrices = mn.make_design_matrix_dm(psr, frot_dummy)
    assert design_matrices.shape == (360, 49, 4), 'Shape of design matrices is not as expected'
    assert np.allclose(design_matrices[0, 0, :], [frot_dummy**-1, 0, 0, (psr.radio_freqs[0][0] / 1400)**-2]), 'First row of first epoch is not as expected'


def test_make_design_matrix():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    frot_dummy = np.random.rand() * 1000
    design_matrices = mn.make_design_matrix(psr, frot_dummy)
    assert design_matrices.shape == (360, 49, 3), 'Shape of design matrices is not as expected'
    assert np.allclose(design_matrices[0, 0, :], [frot_dummy**-1, 0, 0]), 'First row of first epoch is not as expected'

def test_make_covariance_matrix_mb_dm():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    out = mn.make_covariance_matrices_mb(psr, mn.covariance_matrix_mb_dm)
    print(out.params)
    params = {p: np.random.rand() for p in out.params}
    Q = out(params)

    assert Q.shape == (360, 4, 4), 'Shape of Q is not as expected'
    assert out.params == ['B1855+09_process_cov_sigma_dphi', 'B1855+09_process_cov_sigma_df', 'B1855+09_process_cov_sigma_dfdot', 'B1855+09_process_cov_sigma_dm_dphi']

def test_make_covariance_matrix_mb():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    out = mn.make_covariance_matrices_mb(psr, mn.covariance_matrix_mb)
    print(out.params)
    params = {p: np.random.rand() for p in out.params}
    Q = out(params)

    assert Q.shape == (360, 3, 3), 'Shape of Q is not as expected'
    assert out.params == ['B1855+09_process_cov_sigma_dphi', 'B1855+09_process_cov_sigma_df', 'B1855+09_process_cov_sigma_dfdot']

def test_make_measurement_covar():
    psr = mn.MultiBandPulsar.read_feather_pre_process("data/v1p1_de440_pint_bipm2019-B1855+09.feather")
    out = mn.make_measurement_covar(psr, psr.noisedict)
    assert out.shape == (360, 49, 49), 'Shape of measurement matrix is not as expected'