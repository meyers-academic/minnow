import numpy as np
import jax.numpy as jnp
import jax
from . import settings
import discovery as ds
from jax.lax import scan
from copy import deepcopy
from typing import Callable


def predict(x, P, transition, covariance):
    xp = transition @ x
    Pp = transition @ P @ transition.T + covariance
    return xp, Pp


def update(xp, Pp, measurement, design, meas_covar, woodbury=False):
    innov = measurement - (design @ xp)
    if woodbury:
        Ninv = settings.jnp.linalg.inv(meas_covar)
        Pinv = settings.jnp.linalg.inv(Pp)
        innov_covar_inv = Ninv @ (settings.jnp.eye(Ninv.shape[0]) - design @ settings.jnp.linalg.inv(Pinv + design.T @ Ninv @ design) @ design.T @ Ninv)

        gain = (innov_covar_inv @ design @ Pp).T

        x = xp + gain @ innov
        P = (settings.jnp.eye(Pp.shape[0]) - gain @ design) @ Pp
        ll = -0.5 * (-settings.jnp.linalg.slogdet(innov_covar_inv)[1] + innov.T @ innov_covar_inv @ innov + sum(abs(measurement)>0) * jnp.log(2 * jnp.pi))
    else:
        innov_covar = meas_covar + design @ Pp @ design.T
        gain = settings.jnp.linalg.solve(innov_covar, design @ Pp).T
        x = xp + gain @ innov
        P = (jnp.eye(Pp.shape[0]) - gain @ design) @ Pp
        ll = -0.5 * (settings.jnp.linalg.slogdet(innov_covar)[1] + innov.T @ settings.jnp.linalg.solve(innov_covar, innov) + sum(abs(measurement)>0) * jnp.log(2 * jnp.pi))

    return x, P, jnp.atleast_1d(ll)


class KalmanMultiBand:
    def __init__(self, psr, transitions, designs, process_covars, measurement_covars, noisedict=None, woodbury=False):
        self.psr = psr
        # preproc

        self.transitions = transitions
        self.designs = designs
        self.process_covars = process_covars
        self.measurement_covariances = measurement_covars
        params = [f'{psr.name}_x0', f'{psr.name}_P0']
        if isinstance(process_covars, Callable):
            params += process_covars.params
        if isinstance(measurement_covars, Callable):
            params += measurement_covars.params
        if isinstance(designs, Callable):
            params += designs.params
        if isinstance(transitions, Callable):
            params += transitions.params
        self.params = sorted(params)
        self.woodbury=woodbury
    def makefilter(self):
        transitions = self.transitions
        designs = self.designs
        process_covars = self.process_covars
        measurement_covariances = self.measurement_covariances

        residuals = self.psr.residuals
        name = self.psr.name
        def filter(params):
            if isinstance(process_covars, Callable):
                Qs = process_covars(params)
            else:
                Qs = process_covars
            if isinstance(measurement_covariances, Callable):
                Rs = measurement_covariances(params)
            else:
                Rs = measurement_covariances

            if isinstance(designs, Callable):
                Ms = designs(params)
            else:
                Ms = designs
            if isinstance(transitions, Callable):
                Fs = transitions(params)
            else:
                Fs = transitions
            x0 = params[f'{name}_x0']
            P0 = jnp.diag(params[f'{name}_P0'])

            def step(carry, inputs):

                x, P = carry
                transition, Q_i, measurement, design, meas_covar = inputs
                # Prediction
                xpred, Ppred = predict(x, P, transition, Q_i)

                # Update
                x, P, ll = update(xpred, Ppred, measurement, design, meas_covar, woodbury=self.woodbury)

                return (x, P), (x, P, xpred, Ppred, ll)

                # Stack inputs for lax.scan
            inputs = (
                Fs,
                Qs,
                residuals,
                Ms,
                Rs
            )
            carry = (x0, P0)
            carry_final, outputs = scan(step, carry, inputs)

            xreturns, Preturns, xpreds, Ppreds, lls = outputs
            return lls, xreturns, Preturns, xpreds, Ppreds
        # filter.params = self.covariance_func.params +
        filter.params = self.params
        return filter

    def make_forward_backward(self):

        # make forward filter
        filter = self.makefilter()

        def forward_backward(params):
            lls, x, P, xpred, Ppred = filter(params)
            xsmooth = x.copy()
            Psmooth = P.copy()
            n = x.shape[0]

            def smooth_step(carry, t):
                xs, Ps = carry
                F = jnp.eye(x.shape[1])  # placeholder
                # Ppred = F @ P[t] @ F.T   # placeholder
                G = P[t] @ F.T @ jnp.linalg.inv(Ppred[t+1])
                new_x = xs.at[t].set(xs[t] + G @ (xs[t+1] - xpred[t+1]))
                new_P = Ps.at[t].set(Ps[t] + G @ (Ps[t+1] - Ppred[t+1]) @ G.T)
                return (new_x, new_P), None

            (xsmooth, Psmooth), _ = jax.lax.scan(
                smooth_step,
                (xsmooth, Psmooth),
                jnp.arange(n - 2, -1, -1)
            )

            return lls, xsmooth, Psmooth, xpred
        return forward_backward



class KalmanSingleBand:
    def __init__(self, psr, transitions, torques, process_covars, measurement_variances):
        self.psr = psr
        self.transitions = transitions
        self.torques = torques
        self.process_covars = process_covars
        self.measurement_variances = measurement_variances
        params = [f'{psr.name}_x0', f'{psr.name}_P0']
        if isinstance(process_covars, Callable):
            params += process_covars.params
        if isinstance(measurement_variances, Callable):
            params += measurement_variances.params
        if isinstance(transitions, Callable):
            params += transitions.params
        if isinstance(torques, Callable):
            params += torques.params
        self.params = sorted(params)

    def makefilter_nophase(self):
        transitions = self.transitions
        torques = self.torques
        process_covars = self.process_covars
        measurement_variances = self.measurement_variances
        psr = self.psr

        def filter(params):
            if isinstance(process_covars, Callable):
                Qs = process_covars(params)
            else:
                Qs = process_covars
            if isinstance(measurement_variances, Callable):
                Rs = measurement_variances(params)
            else:
                Rs = measurement_variances

            if isinstance(torques, Callable):
                Ts = torques(params)
            else:
                Ts = torques
            if isinstance(transitions, Callable):
                Fs = transitions(params)
            else:
                Fs = transitions

            x0 = jnp.array(params[f'{psr.name}_x0']).reshape((3, 1))
            P0 = jnp.diag(params[f'{psr.name}_P0'])
            # jax.debug.print('P0 {x}', x=P0)
            def step(carry, inputs):
                # jax.debug.print('carry {x}', x=carry)

                x, P = carry
                transition, Q, torque, measurement_variance = inputs
                # Prediction
                xpred, Ppred = predict_nophase(x, P, transition, torque, Q)
                # Update
                x, P, innov_covar, ll = update_nophase(xpred, Ppred, measurement_variance, settings.jnp.array([1, 0, 0]).reshape((1, 3)))
                # round phase, we don't need the full value
                phase_int = settings.jnp.round(x[0, 0])
                x = x.at[0, 0].set(x[0, 0] - phase_int)
                return (x, P), (x, P, xpred, Ppred, innov_covar, ll)
            inputs = (
                Fs,
                Qs,
                Ts,
                Rs
            )
            carry=(x0, P0)
            carry_final, outputs = scan(step, carry, inputs)
            xreturns, Preturns, xpreds, Ppreds, innov_covars, lls = outputs
            return lls, innov_covars, xreturns.squeeze(), Preturns.squeeze(), xpreds.squeeze(), Ppreds.squeeze()
        filter.params = self.params
        return filter

def update_nophase(xp, Pp, dt_err, measurement_jac):

    innov = settings.jnp.round(xp[0, 0]) - xp[0, 0]
    innov_covar = measurement_jac @ Pp @ measurement_jac.T + dt_err * xp[1, 0] ** 2

    gain = (Pp @ measurement_jac.T) / innov_covar
    x = xp + gain * innov

    P = (settings.jnp.eye(3) - gain @ measurement_jac) @ Pp
    ll = -0.5 * (settings.jnp.log(innov_covar) + innov ** 2 / innov_covar + settings.jnp.log(2 * np.pi))
    return x, P, innov_covar, ll

def predict_nophase(x, P, transition, torque, process_covariance):
    xp = transition @ x + jnp.atleast_2d(torque).T
    Pp = transition @ P @ transition.T + process_covariance
    return xp, Pp