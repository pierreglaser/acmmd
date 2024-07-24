from typing import Literal
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from utils import config  # type: ignore

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

def plot_quantity_evolution(
    df,
    y_axis_key,
    groupby_key,
    x_axis_key,
    label_fmt_str,
    use_log_scale_x=False,
    use_log_scale_y=False,
    plot_yerr=True,
    filename_suffix="",
    ax=None,
    f=None,
    plot_legend=True,
    save=True,
    filename_override=None,
    title_override=None,
    ylabel_override=None,
):
    if ax is None:
        assert f is None
        f, ax = plt.subplots()
    else:
        assert f is not None

    val_mean_by_num_samples_and_delta_t = (
        df.groupby(level=["upsample_to", "delta_t"], axis=0)[y_axis_key]
        .mean()
        .to_frame("mean")
    )

    if y_axis_key == "result":
        val_std_by_num_samples_and_delta_t = (
            df.groupby(level=["upsample_to", "delta_t"], axis=0)[y_axis_key]
            .apply(lambda x: np.sqrt(x.mean() * (1 - x.mean()) / x.count()))
            .to_frame("std")
        )
    else:
        val_std_by_num_samples_and_delta_t = (
            df.groupby(level=["upsample_to", "delta_t"], axis=0)[y_axis_key]
            .std()
            .to_frame("std")
        )

    mean_and_std = pd.concat(
        [val_mean_by_num_samples_and_delta_t, val_std_by_num_samples_and_delta_t],
        axis=1,
    )

    with mpl.rc_context(fname="figures.mplstyle"):
        for key, df_ in mean_and_std.groupby(level=groupby_key):
            if plot_yerr:
                ax.errorbar(
                    x=df_.index.get_level_values(x_axis_key),
                    y=df_["mean"],
                    yerr=df_["std"],
                    label=label_fmt_str.format(key),
                    marker="o",
                    markersize=config["markersize"],
                    elinewidth=0.5,
                    capsize=2,
                    capthick=0.5,
                    linewidth=config["linewidth"],
                    linestyle="--",
                )
            else:
                ax.plot(
                    df_.index.get_level_values(x_axis_key),
                    df_["mean"],
                    label=label_fmt_str.format(key),
                    marker="o",
                    markersize=config["markersize"],
                )

        # also set the xtick
        ax.set_xlabel(name_map[x_axis_key], fontsize=config["xlabel_fontsize"])
        if ylabel_override is not None:
            ax.set_ylabel(ylabel_override, fontsize=config["ylabel_fontsize"])
        else:
            ax.set_ylabel(name_map[y_axis_key], fontsize=config["ylabel_fontsize"])

        ax.yaxis.get_offset_text().set_fontsize(config["ytick_fontsize"])

        if plot_legend:
            pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
            h, l = ax.get_legend_handles_labels()
            f.legend(
                h, l,
                loc='lower center', 
                bbox_to_anchor=(0.5, -0.1),
                # bbox_to_anchor=(0., -0.3),
                ncol=5, 
            )
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            # ax.legend(
            #     loc='center right', bbox_to_anchor=(1.35, 0.5),
            #     title=r"$\Delta T$",
            #     title_fontsize=config["legend_title_fontsize"],
            # )

        if use_log_scale_y:
            ax.set_yscale("log", base=10)
            plt.minorticks_off()


        # use scientific notation for the y axis
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


        if use_log_scale_x:
            # min_x, max_x = df.index.get_level_values(x_axis_key).min(), df.index.get_level_values(x_axis_key).max()
            x_vals = df.index.get_level_values(x_axis_key).unique().sort_values()
            ax.set_xticks(x_vals)
            # ax.set_xticklabels([f"{min_x:.0e}", f"{max_x:.0e}"])
            ax.set_xscale("log", base=10)
            # plt.minorticks_off()
        else:
            ax.set_xticks([100, 1000])
            ax.set_xticklabels(["100", "1000"])

        plt.xticks(fontsize=config["xtick_fontsize"])
        plt.yticks(fontsize=config["ytick_fontsize"])

        if title_override is not None:
            f.suptitle(
                # r"$\widehat{D}(\mathbb{P}_{|}, Q_{|}) \mathrm{\,\,between\,\, two\,\,different\,\,ProteinMPNN\,\,models}$",
                title_override,
                fontsize=config["title_fontsize"],
                # pad=20
            )
        else:
            f.suptitle(
                # r"$\widehat{D}(\mathbb{P}_{|}, Q_{|}) \mathrm{\,\,between\,\, two\,\,different\,\,ProteinMPNN\,\,models}$",
                r"$\mathrm{Comparing \,\,two\,\,different\,\,ProteinMPNN\,\,models}$",
                fontsize=config["title_fontsize"],
                # pad=20
            )
        plt.tight_layout()

        synthetic_exp_folder = Path("./synthetic_exps_plots")
        synthetic_exp_folder.mkdir(exist_ok=True)

        if save:
            if filename_override is not None:
                filename = filename_override
            else:
                filename = f"./mmd_cgof_temperature_{y_axis_key}_vs_{x_axis_key}_{filename_suffix}.pdf"
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
        df = pd.read_csv(
            "../results/results_temperature_rebuttals.csv",
            index_col=0,
        )
    elif type_ == "calibration":
        df = pd.read_csv(
            "../results/result_calibration.csv",
            index_col=0,
        )
    else:
        raise ValueError(f"Unknown type {type_}")

    df["biased_val"] = df["val"] + 1 / df["upsample_to"]

    plt.ion()

    df_f = df.loc[df['delta_t'].isin(
        [0., 0.001, 0.01, 0.05, 0.1]
    )]

    df_f = df_f.iloc[:, 4:].set_index(
        ["temperature_true_model", "upsample_to", "delta_t", "torch_seed"]
    )

    combinations = [
        dict(
            x_axis_key="upsample_to",
            y_axis_key="val",
            groupby_key="delta_t",
            label_fmt_str=r"${}$",
        ),
        # dict(
        #     x_axis_key="upsample_to",
        #     y_axis_key="biased_val",
        #     groupby_key="delta_t",
        #     label_fmt_str=r"$\Delta T = {}$",
        # ),
        dict(
            x_axis_key="upsample_to",
            y_axis_key="result",
            groupby_key="delta_t",
            # label_fmt_str=r"$\Delta T = {}$",
            label_fmt_str=r"${}$",
        ),
    ]

    # for temp_ in [0.1, 1]:
    #     slice_ = df_f.loc[temp_]


    #     for kws in combinations:
    #         plot_quantity_evolution(
    #             use_log_scale_x=kws["x_axis_key"] == "upsample_to",
    #             use_log_scale_y=kws["y_axis_key"] == "biased_val",
    #             filename_suffix=f"temperature_{temp_}_type_{type_}",
    #             **kws,
    #         )


    for temp_ in [0.1, 1][:1]:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        slice_ = df_f.loc[temp_]
        for kws, ax in zip(combinations, [ax1, ax2]):
            plot_quantity_evolution(
                df=slice_,
                use_log_scale_x=(type_ == "cgof"),
                use_log_scale_y=kws["y_axis_key"] == "biased_val",
                filename_suffix="",
                ax=ax,
                f=f,
                save=True if ax is ax2 else False,
                filename_override=f"temperature_{temp_}_type_{type_}.pdf",
                plot_legend=(ax is ax2),
                title_override=r"$\mathrm{Calibration\,\, Mismatch \,\, for\,\, two\,\,different\,\,ProteinMPNN\,\,models}$" if type_ == "calibration" else None,
                ylabel_override=r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|Q_{|}}, Q_{|})$" if type_ == "calibration" else None,
                **kws,
            )
