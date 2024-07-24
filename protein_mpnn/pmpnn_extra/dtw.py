# pyright: basic, reportPrivateUsage=false
# Inspired by https://github.com/khdlr/softdtw_jax/blob/main/softdtw_jax/softdtw_jax.py

from typing import Any, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
from calibration.utils import median_median_heuristic_discrete
from flax import struct
from jax.scipy.special import logsumexp
from kwgflows.base import base_kernel
from kwgflows.pytypes import Array, Scalar
from kwgflows.rkhs.kernels import MedianHeuristicKernel, gaussian_kernel
from typing_extensions import Self, Type


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=-jnp.inf)


class DTW(struct.PyTreeNode):
    gamma: float = 0.1
    # kernel: base_kernel[Array] = ModifiedGaussianKernel(sigma=1.0)
    kernel: base_kernel[Array] = gaussian_kernel(sigma=1.0)

    def __call__(self, prediction, target, log_space=False):
        log_K = jnp.log(self.kernel.make_gram_matrix(prediction, target))
        # wlog: H >= W
        if log_K.shape[0] < log_K.shape[1]:
            int_ = prediction
            prediction = target
            target = int_

            log_K = log_K.T
        H, W = log_K.shape

        rows = []
        for row in range(H):
            rows.append(pad_inf(log_K[row], row, H - row - 1))

        # shape H + W - 1, H
        model_matrix = jnp.stack(rows, axis=1)

        init = (
            pad_inf(model_matrix[0], 1, 0),
            pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0),
        )

        def dtw_scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = self.gamma * logsumexp(
                jnp.stack([diagonal, right, down], axis=-1) / self.gamma, axis=-1
            )
            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling
        carry = init
        ys = []
        for i, row in enumerate(model_matrix[2:]):
            carry, y = dtw_scan_step(carry, row)
            ys.append(y)
        ys = jnp.stack(ys, axis=0)

        # in case prediction/target are padded with nans to indicate
        # end of sequences (allows for efficient vectorization)
        # XXX: if variable number of nans, take the dimension with
        # the most nans
        delta_H = H - jnp.max(jnp.sum(jnp.isfinite(prediction), axis=0))
        delta_W = W - jnp.max(jnp.sum(jnp.isfinite(target), axis=0))

        _, ys = jax.lax.scan(dtw_scan_step, init, model_matrix[2:], unroll=4)
        # coordinate system of ys is i=antidiag no, j=pos on antidiag.
        # yields col_idx=j, row_idx=i-j
        # for col_idx=H, row_idx=W, yields j=H, i=H+W
        # since
        # - number of antidiagonals H + W - 1
        # - two antidiagonals are already in init
        # - 0 indexed
        # must subtract 4,1
        # i=H+W-4, j=H-1
        # each column padded with one inf on the left
        # -> i=H+W-4, j=H
        # when n and m are padded with nans, yields
        # i=(H-delta_H)+(W-delta_W)-4, j=(H-delta_H)
        log_ret = ys[(H - delta_H) + (W - delta_W) - 4, (H - delta_H)]
        if log_space:
            return log_ret
        else:
            return jnp.exp(log_ret)
