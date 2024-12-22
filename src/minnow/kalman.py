import numpy as np
import jax.numpy as jnp
import jax
from . import config
import discovery as ds
from jax.lax import scan


def predict(x, P, transition, covariance):
    xp = transition @ x
    Pp = transition @ P @ transition.T + covariance
    return xp, Pp


def update(xp, Pp, measurement, design, meas_covar):
    innov = measurement - (design @ xp)
    inn_cov = design @ Pp @ design.T + meas_covar
    gain = jnp.linalg.solve(inn_cov, design @ Pp).T  # Efficient solve
    x = xp + gain @ innov
    P = (jnp.eye(Pp.shape[0]) - gain @ design) @ Pp
    ll = -0.5 * (jnp.linalg.slogdet(inn_cov)[1] + innov.T @ jnp.linalg.solve(inn_cov, innov) + sum(abs(measurement)>0) * jnp.log(2 * jnp.pi))
    return x, P, jnp.atleast_1d(ll)


class Kalman_varQ:
    def __init__(self, psr, transitions, designs, covariance_func, measurement_covars, noisedict=None):
        self.psr = psr
        # preproc

        self.transitions = transitions
        self.designs = designs
        self.covariance_func = covariance_func
        self.measurement_covariances = measurement_covars
        self.params = sorted(covariance_func.params + [f'{psr.name}_x0', f'{psr.name}_P0'])

    def makefilter(self):
        transitions = self.transitions
        designs = self.designs
        covariance_func = self.covariance_func
        measurement_covariances = self.measurement_covariances

        residuals = self.psr.residuals
        name = self.psr.name
        print("HI")
        def filter(params):
            Qs = covariance_func(params)
            x0 = params[f'{name}_x0']
            print('CFG', config)
            P0 = jnp.diag(params[f'{name}_P0'])

            def step(carry, inputs):
                    x, P = carry
                    transition, Q_i, measurement, design, meas_covar = inputs
                    # Prediction
                    xpred, Ppred = predict(x, P, transition, Q_i)

                    # Update
                    x, P, ll = update(xpred, Ppred, measurement, design, meas_covar)

                    return (x, P), (x, P, xpred, ll)

                # Stack inputs for lax.scan
            inputs = (
                transitions,
                Qs,
                residuals,
                designs,
                measurement_covariances
            )
            carry = (x0, P0)
            carry_final, outputs = scan(step, carry, inputs)

            xreturns, Preturns, xpreds, lls = outputs
            return lls, xreturns, Preturns, xpreds
        # filter.params = self.covariance_func.params +
        filter.params = self.params
        return filter