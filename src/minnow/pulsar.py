import discovery as ds
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from enterprise.pulsar import Pulsar as epPulsar


class MultiBandPulsar(ds.Pulsar):
    """MultiBandPulsar -- A class for handling pulsar data with multiple frequency channels
    at given TOAs.
    """
    def __init__(self, toas, residuals, radio_frequencies, toaerrs, backend_flags, Mmat, noisedict=None, name='psr', cutidxs=[6, 7]):

        bins = ds.quantize(toas)
        ubins, counts = np.unique(bins, return_counts=True) # number of TOAs per bin
        maxcounts = np.max(counts)
        self.toas = np.array([np.zeros(maxcounts) for ii in range(counts.size)])
        self.residuals = np.array([np.zeros(maxcounts) for ii in range(counts.size)])
        self.toaerrs = np.array([np.ones(maxcounts) for ii in range(counts.size)])
        self.radio_freqs = np.array([np.ones(maxcounts) for ii in range(counts.size)]) * 1400
        self.masks = np.array([np.zeros(maxcounts) for ii in range(counts.size)])
        counts_prep = np.insert(counts, 0, 0)
        self.mean_toas = []
        self.counts = counts

        # self.backend_flags = backend_flags
        self.backend_flags = []

        # cut down Mmat
        self.name = name

        self.noisedict = noisedict

    # get indexes associated with each bin for pulling out TOAs and putting
    # them into padded arrays
        idxs = np.cumsum(counts_prep)
        Mmat = np.delete(Mmat, cutidxs, axis=1)
        Mmat, _, _ = np.linalg.svd(Mmat, full_matrices=False)
        self.Mmats = np.zeros((counts.size, maxcounts, Mmat.shape[1]))

        for ii,c in enumerate(counts):
            self.residuals[ii][:c] = residuals[idxs[ii]:idxs[ii+1]]
            self.toas[ii][:c] = toas[idxs[ii]:idxs[ii+1]]
            self.mean_toas.append(np.mean(toas[idxs[ii]:idxs[ii+1]]))
            self.toaerrs[ii][:c] = toaerrs[idxs[ii]:idxs[ii+1]]
            self.radio_freqs[ii][:c] = radio_frequencies[idxs[ii]:idxs[ii+1]]
            self.masks[ii][:c] = np.ones(c)
            # assume we can just take one column of the Mmatrix
            # for each epoch.
            # We should probably test this, though.
            # Do not take F and F1 columns, which we assume are first.
            # self.Mmat.append(np.mean(Mmat[idxs[ii]:idxs[ii+1]], axis=0))
            self.Mmats[ii, :c, :] = Mmat[idxs[ii]:idxs[ii+1]]

            if np.unique(backend_flags[idxs[ii]:idxs[ii+1]]).size > 1:
                logger.warning("There are multiple backend flags in a single epoch")
                # raise ValueError("There are multiple backend flags in a single epoch")
            self.backend_flags.append(backend_flags[idxs[ii]])

        self.mean_toas = np.insert(self.mean_toas, 0, self.mean_toas[0]-1)
        self.toa_diffs = np.diff(self.mean_toas)

    @classmethod
    def read_feather_pre_process(cls, fname):
        return cls.from_discovery(ds.Pulsar.read_feather(fname))

    @classmethod
    def from_discovery(cls, ds_psr):

        if hasattr(ds_psr, 'noisedict'):
            noisedict = ds_psr.noisedict
        else:
            noisedict = None
        return cls(ds_psr.toas, ds_psr.residuals,
                   ds_psr.freqs, ds_psr.toaerrs,
                   ds_psr.backend_flags, ds_psr.Mmat,
                   noisedict=noisedict, name=ds_psr.name)

def standard_cut(Mmat, fitpars):
    idx = 0
    cuts = []
    for fp in fitpars:
        if "F0"==fp:
            cuts.append(idx)
        if "F1"==fp:
            cuts.append(idx)
        if "DMX" in fp:
            cuts.append(idx)
        if "DMJUMP" in fp:
            cuts.append(idx)
        idx += 1
    return np.delete(Mmat, cuts, axis=1)


class SingleBandPulsar(ds.Pulsar):
    """Single -- A class for handling pulsar data with a single frequency channels
    at given TOA.
    """
    def __init__(self, toas, residuals, radio_frequencies,
                 toaerrs, backend_flags, Mmat, fitpars,
                 noisedict=None, name='psr'):

        self.toas = toas
        self.toaerrs = toaerrs
        self.residuals = residuals
        self.radio_freqs = radio_frequencies
        self.backend_flags = backend_flags
        self.fitpars = fitpars

        self.toa_diffs = np.diff(toas)
        self.toa_diff_errors = np.sqrt(toaerrs[1:]**2 + toaerrs[:-1]**2)

        # cut up Mmat
        # Mmat = cutfunc(Mmat, fitpars)
        self.Mmat = Mmat
        if noisedict:
            self.noisedict = noisedict
        self.name = name


    @classmethod
    def read_par_tim(cls, p, t, **kwargs):
        return cls.from_enterprise(epPulsar(str(p), str(t), **kwargs))

    @classmethod
    def from_enterprise(cls, ds_psr):

        if hasattr(ds_psr, 'noisedict'):
            noisedict = ds_psr.noisedict
        else:
            noisedict = None
        return cls(ds_psr.toas, ds_psr.residuals,
                   ds_psr.freqs, ds_psr.toaerrs,
                   ds_psr.backend_flags, ds_psr.Mmat, ds_psr.fitpars,
                   noisedict=noisedict, name=ds_psr.name)