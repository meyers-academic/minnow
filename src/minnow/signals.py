import numpy as np
from . import config as cfg
import inspect
import jax


def make_residual_tracking_transition_matrix_dm(mb_psr):
    """Make transition matrix for residual tracking with DM as a state variable.

    Parameters
    ----------
    mb_psr : minnow.MultiBandPulsar
        _description_

    Returns
    -------
    transition_matrices : jnp.array
        transition matrices with dimensions (n_epochs, n_states, n_states)
    """
    transition_matrices = cfg.jnp.array([[np.ones(mb_psr.toa_diffs.size), mb_psr.toa_diffs, mb_psr.toa_diffs**2 / 2, np.zeros(mb_psr.toa_diffs.size)],
                                         [np.zeros(mb_psr.toa_diffs.size), np.ones(mb_psr.toa_diffs.size), mb_psr.toa_diffs, np.zeros(mb_psr.toa_diffs.size)],
                                         [np.zeros(mb_psr.toa_diffs.size), np.zeros(mb_psr.toa_diffs.size), np.ones(mb_psr.toa_diffs.size), np.zeros(mb_psr.toa_diffs.size)],
                                         [np.zeros(mb_psr.toa_diffs.size), np.zeros(mb_psr.toa_diffs.size), np.zeros(mb_psr.toa_diffs.size), np.ones(mb_psr.toa_diffs.size)]])
    transition_matrices = cfg.jnp.swapaxes(cfg.jnp.swapaxes(transition_matrices, 0, 2), 1, 2)
    return transition_matrices

def make_residual_tracking_transition_matrix(mb_psr):
    """Make transition matrix for residual tracking with DM as a state variable.

    Parameters
    ----------
    mb_psr : minnow.MultiBandPulsar
        _description_

    Returns
    -------
    transition_matrices : jnp.array
        transition matrices with dimensions (n_epochs, n_states, n_states)
    """

    transition_matrices = cfg.jnp.array([[np.ones(mb_psr.toa_diffs.size), mb_psr.toa_diffs, mb_psr.toa_diffs**2 / 2],
                                         [np.zeros(mb_psr.toa_diffs.size), np.ones(mb_psr.toa_diffs.size), mb_psr.toa_diffs],
                                         [np.zeros(mb_psr.toa_diffs.size), np.zeros(mb_psr.toa_diffs.size), np.ones(mb_psr.toa_diffs.size)]])
    transition_matrices = cfg.jnp.swapaxes(cfg.jnp.swapaxes(transition_matrices, 0, 2), 1, 2)
    return transition_matrices

def make_design_matrix_dm(mb_psr, rotation_frequency):
    """Make design matrix for residual tracking with DM as a state variable.

    Parameters
    ----------
    mb_psr : minnow.MultiBandPulsar
        _description_

    Returns
    -------
    design_matrices : jnp.array
        design matrices with dimensions (n_epochs, max toas per epoch, 4)
    """
    maxcounts = np.max(mb_psr.counts)

    Umats = [np.zeros((maxcounts, 4)) for c in mb_psr.counts]


    # prepend counts
    counts_prep = np.insert(mb_psr.counts, 0, 0)

    # get indexes associated with each bin for pulling out TOAs and putting
    # them into padded arrays
    idxs = np.cumsum(counts_prep)

    # design only goes up to the value we need
    for um, c, rf in zip(Umats, mb_psr.counts, mb_psr.radio_freqs):
        um[:c, 0] = np.ones(c) / rotation_frequency
        um[:c, -1] = (1400 * np.ones(c) / rf[:c])**2

    return np.array(Umats)

def make_design_matrix(mb_psr, rotation_frequency):
    """Make design matrix for residual tracking

    Parameters
    ----------
    mb_psr : minnow.MultiBandPulsar
        _description_

    Returns
    -------
    design_matrices : jnp.array
        design matrices with dimensions (n_epochs, max toas per epoch, 3)
    """
    maxcounts = np.max(mb_psr.counts)

    Umats = [np.zeros((maxcounts, 3)) for c in mb_psr.counts]


    # prepend counts
    counts_prep = np.insert(mb_psr.counts, 0, 0)

    # get indexes associated with each bin for pulling out TOAs and putting
    # them into padded arrays
    idxs = np.cumsum(counts_prep)

    # design only goes up to the value we need
    for um, c, rf in zip(Umats, mb_psr.counts, mb_psr.radio_freqs):
        um[:c, 0] = np.ones(c) / rotation_frequency

    return np.array(Umats)

def covariance_matrix_mb_dm(toa_diffs, sigma_dphi, sigma_df, sigma_dfdot, sigma_dm_dphi):
    size = toa_diffs.size
    Q = cfg.jnparray([[toa_diffs * (sigma_df**2 * toa_diffs**2 / 3 + sigma_dfdot**2 * toa_diffs**4 / 20 + sigma_dphi**2),
                            toa_diffs**2 * (4 * sigma_df**2 + sigma_dfdot**2 * toa_diffs**2) / 8,
                            sigma_dfdot**2 * toa_diffs**3 / 6, cfg.jnp.zeros(size)],
                            [toa_diffs**2 * (4 * sigma_df**2 + sigma_dfdot**2 * toa_diffs**2) / 8,
                            toa_diffs * (sigma_df**2 + sigma_dfdot**2 * toa_diffs**2 / 3),
                            sigma_dfdot**2 * toa_diffs**2 / 2, cfg.jnp.zeros(size)],
                            [sigma_dfdot**2 * toa_diffs**3 / 6,
                            sigma_dfdot**2 * toa_diffs**2 / 2,
                            sigma_dfdot**2 * toa_diffs, cfg.jnp.zeros(size)],
                        [cfg.jnp.zeros(size),cfg.jnp.zeros(size),cfg.jnp.zeros(size),sigma_dm_dphi**2*toa_diffs]])
    return cfg.jnp.swapaxes(cfg.jnp.swapaxes(Q, 0, 2), 1, 2)

def covariance_matrix_mb(toa_diffs, sigma_dphi, sigma_df, sigma_dfdot):
    size = toa_diffs.size
    Q = cfg.jnparray([[toa_diffs * (sigma_df**2 * toa_diffs**2 / 3 + sigma_dfdot**2 * toa_diffs**4 / 20 + sigma_dphi**2),
                            toa_diffs**2 * (4 * sigma_df**2 + sigma_dfdot**2 * toa_diffs**2) / 8,
                            sigma_dfdot**2 * toa_diffs**3 / 6],
                            [toa_diffs**2 * (4 * sigma_df**2 + sigma_dfdot**2 * toa_diffs**2) / 8,
                            toa_diffs * (sigma_df**2 + sigma_dfdot**2 * toa_diffs**2 / 3),
                            sigma_dfdot**2 * toa_diffs**2 / 2],
                            [sigma_dfdot**2 * toa_diffs**3 / 6,
                            sigma_dfdot**2 * toa_diffs**2 / 2,
                            sigma_dfdot**2 * toa_diffs]])
    return cfg.jnp.swapaxes(cfg.jnp.swapaxes(Q, 0, 2), 1, 2)



def make_covariance_matrices_mb(mb_psr, cov_function, name='process_cov', common=[]):
    argspec = inspect.getfullargspec(cov_function)
    argmap = [arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else
                  f'{mb_psr.name}_{name}_{arg}' for arg in argspec.args if arg not in ['toa_diffs']]
    toa_diffs = mb_psr.toa_diffs

    def make_covar(params):
        pass
        Q = cov_function(toa_diffs, *[params[arg] for arg in argmap])
        return Q
    make_covar.params = argmap
    return make_covar

def selection_backend_flags(psr):
    return psr.backend_flags

def make_measurement_covar(psr, noisedict={}, tnequad=False, selection=selection_backend_flags):
    """Make measurement covariance matrix for pulsar data.

    Parameters
    ----------
    psr : mn.MultiBandPulsar
        Pulsar data
    noisedict : dict, optional
        white noise dictionary, by default {}
    tnequad : bool, optional
        equad convention, by default False
    selection : Callable, optional
        for making selections, by default selection_backend_flags, not implemented

    Returns
    -------
    measurement_covars : jnp.array
        measurement covariance matrices
    """

    efacs = [noisedict[f'{psr.name}_{be}_efac'] * mask for be, mask in zip(psr.backend_flags, psr.masks)]
    if tnequad:
        equads = [10**noisedict[f'{psr.name}_{be}_log10_tnequad'] * mask for be, mask in zip(psr.backend_flags, psr.masks)]
    else:
        equads = [10**noisedict[f'{psr.name}_{be}_log10_t2equad'] * mask for be, mask in zip(psr.backend_flags, psr.masks)]
    ecorrs = [10**noisedict[f'{psr.name}_{be}_log10_ecorr'] * mask for be, mask in zip(psr.backend_flags, psr.masks)]

    efacs  = np.array(efacs)
    equads = np.array(equads)
    ecorrs = np.array(ecorrs)

    efacs[efacs==0] = 1.
    equads[equads==0] = 1.
    ecorrs[ecorrs==0] = 1.
    nepochs = len(ecorrs)
    maxcounts = np.max(psr.counts)
    if tnequad:
        meas_covars = jax.vmap(lambda err, efac, equad, ecorr: (cfg.jnp.diag(efac**2 * err**2 + equad**2) + np.ones((maxcounts, maxcounts)) * (ecorr**2)[:, None]))(psr.toaerrs, efacs, equads, ecorrs)
    else:
        meas_covars = jax.vmap(lambda err, efac, equad, ecorr: (cfg.jnp.diag(efac**2 * (err**2 + equad**2)) + np.ones((maxcounts, maxcounts)) * (ecorr**2)[:, None]))(psr.toaerrs, efacs, equads, ecorrs)

    return meas_covars
