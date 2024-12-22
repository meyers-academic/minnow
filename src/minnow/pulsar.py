import discovery as ds
import numpy as np


class MultiBandPulsar(ds.Pulsar):
    """MultiBandPulsar -- A class for handling pulsar data with multiple frequency channels
    at given TOAs.
    """
    def __init__(self, toas, residuals, radio_frequencies, toaerrs, backend_flags, noisedict=None, name='psr', Mmat=None):

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

        if Mmat is None:
            self.Mmat = Mmat

        self.name = name

        self.noisedict = noisedict

    # get indexes associated with each bin for pulling out TOAs and putting
    # them into padded arrays
        idxs = np.cumsum(counts_prep)

        for ii,c in enumerate(counts):
            self.residuals[ii][:c] = residuals[idxs[ii]:idxs[ii+1]]
            self.toas[ii][:c] = toas[idxs[ii]:idxs[ii+1]]
            self.mean_toas.append(np.mean(toas[idxs[ii]:idxs[ii+1]]))
            self.toaerrs[ii][:c] = toaerrs[idxs[ii]:idxs[ii+1]]
            self.radio_freqs[ii][:c] = radio_frequencies[idxs[ii]:idxs[ii+1]]
            self.masks[ii][:c] = np.ones(c)

            if np.unique(backend_flags[idxs[ii]:idxs[ii+1]]).size > 1:
                raise ValueError("There are multiple backend flags in a single epoch")
            self.backend_flags.append(backend_flags[idxs[ii]])

        self.mean_toas = np.insert(self.mean_toas, 0, self.mean_toas[0]-1)
        self.toa_diffs = np.diff(self.mean_toas)

    @classmethod
    def read_feather_pre_process(cls, fname, pass_mmat=False):
        return cls.from_discovery(ds.Pulsar.read_feather(fname), pass_mmat=pass_mmat)

    @classmethod
    def from_discovery(cls, ds_psr, pass_mmat=False):
        if pass_mmat:
            Mmat = ds_psr.Mmat
        else:
            Mmat = None

        if hasattr(ds_psr, 'noisedict'):
            noisedict = ds_psr.noisedict
        else:
            noisedict = None
        return cls(ds_psr.toas, ds_psr.residuals,
                   ds_psr.freqs, ds_psr.toaerrs, ds_psr.backend_flags, noisedict=noisedict, Mmat=Mmat, name=ds_psr.name)
