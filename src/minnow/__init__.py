#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Minnow"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from .signals import *
from .config import *
from .kalman import *
from .pulsar import *

__version__ = "0.1"