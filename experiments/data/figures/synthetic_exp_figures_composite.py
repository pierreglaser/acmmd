from typing import Literal
from jax import vmap
from kwgflows.rkhs.kernels import gaussian_kernel
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from utils import config  # type: ignore
import jax.numpy as jnp
import sys

sys.path.append("..")
from synthetic_exp import SequenceDistribution, HammingExpKernel
from synthetic_exp_calibration import ExpMMDKernelSequenceModel

name_map = {
    "upsample_to": r"$\mathrm{Num.\,\,Samples}$",
    "delta_t": r"Temperature Difference",
    # "val": r"$\\widehat{D}(\\mathbb P_{|}, Q_{|})$",
    # "biased_val": r"$\\widehat{D}(\\mathbb P_{|}, Q_{|})$ (biased)",
    "val": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$",
    "biased_val": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$ (biased)",
    # "result": r"MMD-CGOF Test Rejection rate",
    "result": r"Rejection rate",
}

plt.style.use("figures.mplstyle")

def acmmd_noncomposite(p, p_p, delta, lambda_):
    ret = 2 * delta**2

    ret *= 1 - jnp.exp(-lambda_)
    ret *= (1 - 2 * p) * (1 - 2 * p_p)
    ret /= 1 - 4 * p * p_p * (1 + jnp.exp(-lambda_)) / 2
    ret *= (
        (2 * p * jnp.exp(-lambda_)) / (1 - 2 * p * jnp.exp(-lambda_))
        + (2 * p_p * jnp.exp(-lambda_)) / (1 - 2 * p_p * jnp.exp(-lambda_))
        + 1
    )
    return ret


def acmmd_composite(p1, p2, delta, lambda_, kx, num_values):
    ps = jnp.linspace(p1, p2, num_values)
    closed_form_mmd_value = 0
    closed_form_mmd_values = vmap(vmap(acmmd_noncomposite, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))(
        ps, ps, delta, lambda_
    )
    kxx = kx.make_gram_matrix(ps, ps)
    closed_form_mmd_value = jnp.sum(closed_form_mmd_values * kxx) / num_values**2
    return closed_form_mmd_value


def acmmd_composite_calibration(p1, p2, delta, lambda_, kx, num_values):
    ps = jnp.linspace(p1, p2, num_values)
    models = vmap(lambda p: SequenceDistribution(p, delta, autopad=True, max_length=20))(ps)

    closed_form_mmd_value = 0
    closed_form_mmd_values = vmap(vmap(acmmd_noncomposite, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))(
        ps, ps, delta, lambda_
    )
    kxx = kx.make_gram_matrix(models, models)
    closed_form_mmd_value = jnp.sum(closed_form_mmd_values * kxx) / num_values**2
    return closed_form_mmd_value



def plot_unbiasedness(
    df,
    ax,
    # groupby_key="delta",
    p1 = 0.3,
    p2 = 0.45,
    num_values=5,
    lambda_ = 1.0,
    type_ = "cgof"
):
    df = df.set_index(
        ["num_samples", "delta", "p1", "p2", "num_values", "lambda_", "test_idx"],
        drop=True,
    )

    df = df.xs(
        (p1, p2, num_values, lambda_), level=["p1", "p2", "num_values", "lambda_"]
    )

    mean_by_num_samples = df.groupby(level=["num_samples", "delta"]).acmmd.mean().to_frame("mean")
    std_by_num_samples = df.groupby(level=["num_samples", "delta"]).acmmd.std().to_frame("std")

    mean_and_std = pd.concat([mean_by_num_samples, std_by_num_samples], axis=1)


    with mpl.rc_context(fname="figures.mplstyle"):
        for delta, df_ in mean_and_std.groupby(level="delta"):
            if delta != 0.3:
                continue

            if type_ == "cgof":
                kx = gaussian_kernel.create()
                acmmd_val = acmmd_composite(p1, p2, delta, lambda_, kx, num_values)
            else:
                k_y = HammingExpKernel(sigma=1 / lambda_)
                kx = ExpMMDKernelSequenceModel.create(ground_space_kernel=k_y, sigma=1.0)
                acmmd_val = acmmd_composite_calibration(p1, p2, delta, lambda_, kx, num_values)

            ax.axhline(
                acmmd_val,
                color="black",
                # label=r"$\mathrm{True\,\,ACMMD}(\mathbb{P}_{|}, Q_{|}), T=" + str(delta) + "$",
                label=r"$\mathrm{ACMMD}(\mathbb{P}_{|}, Q_{|})$",
                linewidth=config["linewidth"],
            )
            ax.errorbar(
                x=df_.index.get_level_values("num_samples"),
                y=df_["mean"],
                yerr=df_["std"],
                # label=r"$\widehat{\mathrm{ACMMD}}_u(\mathbb{P}_{|}, Q_{|}), T=" + str(delta) + "$",
                label=r"$\widehat{\mathrm{ACMMD}}_u(\mathbb{P}_{|}, Q_{|})$",
                marker="o",
                markersize=config["markersize"],
                elinewidth=0.5,
                capsize=2,
                capthick=0.5,
                linewidth=config["linewidth"],
                linestyle="--",
            )

            ax.errorbar(
                x=df_.index.get_level_values("num_samples"),
                y=df_["mean"] + 1 / np.sqrt(df_.index.get_level_values("num_samples")),
                yerr=df_["std"],
                # label=r"$\widehat{\mathrm{ACMMD}}_b(\mathbb{P}_{|}, Q_{|}), T=" + str(delta) + "$",
                label=r"$\widehat{\mathrm{ACMMD}}_b(\mathbb{P}_{|}, Q_{|})$",
                marker="o",
                markersize=config["markersize"],
                elinewidth=0.5,
                capsize=2,
                capthick=0.5,
                linewidth=config["linewidth"],
                linestyle="--",
            )

        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        h, l = ax.get_legend_handles_labels()
        f.legend(
            h, l,
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.2),
            # bbox_to_anchor=(0., -0.3),
            ncol=2, 
        )
        if type_ == "cgof":
            ax.set_title(
                r"$\mathrm{ACMMD}(\mathbb{P}_{|}, Q_{|})$ Estimates"
            )
        else:
            ax.set_title(
                r"$\mathrm{ACMMD}(\mathbb{P}_{|Q}, Q_{|})$ Estimates"
            )

        ax.set_xlabel(name_map["upsample_to"])

        plt.tight_layout()

        synthetic_exp_folder = Path("./synthetic_exps_plots_rebuttals")
        synthetic_exp_folder.mkdir(exist_ok=True)

        filename = f"./synthetic_mmd_cgof_unbiasedness_{type_}_composite.pdf"
        f.savefig(
            synthetic_exp_folder / filename,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )

def plot_power(
    df,
    ax,
    # groupby_key="delta",
    p1 = 0.3,
    p2 = 0.45,
    num_values=5,
    lambda_ = 1.0,
    type_ = "cgof"
):
    df = df.set_index(
        ["num_samples", "delta", "p1", "p2", "num_values", "lambda_", "test_idx"],
        drop=True,
    )

    df = df.xs(
        (p1, p2, num_values, lambda_), level=["p1", "p2", "num_values", "lambda_"]
    )


    mean_by_num_samples = df.groupby(level=["num_samples", "delta"]).reject_h0.mean().to_frame("mean")
    # std_by_num_samples = df.groupby(level=["num_samples", "delta"]).acmmd.std().to_frame("std")

    std_by_num_samples = (
        df.groupby(level=["num_samples", "delta"], axis=0).reject_h0
        .apply(lambda x: np.sqrt(x.mean() * (1 - x.mean()) / x.count()))
        .to_frame("std")
    )

    mean_and_std = pd.concat([mean_by_num_samples, std_by_num_samples], axis=1)


    with mpl.rc_context(fname="figures.mplstyle"):
        for delta, df_ in mean_and_std.groupby(level="delta"):
            ax.errorbar(
                x=df_.index.get_level_values("num_samples"),
                y=df_["mean"],
                yerr=df_["std"],
                label=r"$\Delta p=" + str(delta) + "$",
                marker="o",
                markersize=config["markersize"],
                elinewidth=0.5,
                capsize=2,
                capthick=0.5,
                linewidth=config["linewidth"],
                linestyle="--",
            )

        ax.axhline(
            0.05,
            color="black",
            # label=r"$\mathrm{True\,\,ACMMD}(\mathbb{P}_{|}, Q_{|}), T=" + str(delta) + "$",
            label=r"$\alpha=0.05$",
            linewidth=config["linewidth"],
        )


        pos = ax.get_position()
        h, l = ax.get_legend_handles_labels()
        f.legend(
            h, l,
            loc='lower center', 
            # bbox_to_anchor=(0.5, -0.1),
            bbox_to_anchor=(0.5, -0.2),
            ncol=3, 
        )

        if type_ == "cgof":
            ax.set_title(
                r"$\mathrm{ACMMD}(\mathbb{P}_{|}, Q_{|})$ Test: Rejection Rate"
            )
        else:
            ax.set_title(
                r"$\mathrm{ACMMD}(\mathbb{P}_{|Q}, Q_{|})$ Test: Rejection Rate"
            )

        ax.set_xlabel(name_map["upsample_to"])

        plt.tight_layout()

        synthetic_exp_folder = Path("./synthetic_exps_plots_rebuttals")
        synthetic_exp_folder.mkdir(exist_ok=True)

        filename = f"./synthetic_mmd_cgof_power_{type_}_composite.pdf"
        f.savefig(
            synthetic_exp_folder / filename,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="cgof")  # cgof or calibration

if __name__ == "__main__":
    args = parser.parse_args()
    type_ = args.type

    if type_ == "cgof":
        df = pd.read_csv("../results/results_synthetic_composite.csv", index_col=0)
    else:
        df = pd.read_csv("../results/results_synthetic_calibration_composite.csv", index_col=0)

    unbiasedness_df = df.loc[~df.median_heuristic]
    f, ax = plt.subplots()
    plot_unbiasedness(unbiasedness_df, ax, type_=type_)

    power_df = df.loc[~df.median_heuristic]
    f, ax = plt.subplots()
    plot_power(power_df, ax, type_=type_)
