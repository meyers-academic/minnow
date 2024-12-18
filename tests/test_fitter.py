#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations

from minnow import fitter

class TestFitter:
    # fake data for fitting
    fake_xvals = np.arange(10)
    fake_yval_errors = 0.1 * np.ones(10)
    fake_noise = np.random.randn(10) * 0.1

    # fake data with noise
    fake_yvals = np.sin(2 * np.pi * fake_xvals / 10) + fake_noise

    def test_object(self):
        fitter.


