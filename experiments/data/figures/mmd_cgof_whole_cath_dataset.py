"""
Plot the evolution of the MMD between ProteinMPNN and the true CATH
distributinon of proteins as a function of the model temperature
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import argparse
from utils import config  # type: ignore



name_map = {
    "val": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$",
    "val_cgof": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$",
    "val_calibration": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|Q_{|}}, Q_{|})$",
    "temperature": r"$\mathrm{Temperature}$",
    "num_samples": r"$\mathrm{Num.\,\,Samples}$",
}

plt.style.use("figures.mplstyle")



def plot_sweep(
    df, sweep_quantity, line_sweep_quantity=None,
    type_="cgof",
    ax=None,
    f=None,
    filename_override=None,
    save=False,
    **fixed_params
):
    if ax is None:
        assert f is None
        f, ax = plt.subplots()
    else:
        assert f is not None

    for k, v in fixed_params.items():
        df = df.loc[df[k] == v]

    if line_sweep_quantity is None:
        avg_mmd_by_quantity = df.groupby(sweep_quantity)['val'].mean()
        std_mmd_by_quantity = df.groupby(sweep_quantity)['val'].std()
    else:
        avg_mmd_by_quantity = df.groupby([sweep_quantity, line_sweep_quantity])['val'].mean().unstack()
        std_mmd_by_quantity = df.groupby([sweep_quantity, line_sweep_quantity])['val'].std().unstack()



    with mpl.rc_context(fname="figures.mplstyle"):
        if line_sweep_quantity is not None:
            for c in avg_mmd_by_quantity.columns:
                ax.errorbar(
                    x=avg_mmd_by_quantity.index,
                    y=avg_mmd_by_quantity[c],
                    yerr=std_mmd_by_quantity[c],
                    marker="o",
                    markersize=config["markersize"],
                    linewidth=config["linewidth"],
                    label=c,
                )
            ax.legend()
        else:
            ax.errorbar(
                x=avg_mmd_by_quantity.index,
                y=avg_mmd_by_quantity,
                yerr=std_mmd_by_quantity,
                marker="o",
                markersize=config["markersize"],
                linewidth=config["linewidth"],
            )

        ax.set_xlabel(name_map[sweep_quantity], fontsize=config['xlabel_fontsize'])
        if type_ == "cgof":
            ax.set_ylabel(name_map["val_cgof"], fontsize=config['ylabel_fontsize'])
        elif type_ == "calibration":
            ax.set_ylabel(name_map["val_calibration"], fontsize=config['ylabel_fontsize'])

        ax.yaxis.get_offset_text().set_fontsize(config['ytick_fontsize'])
        # ax.legend()


        # ax.set_ylim(bottom=3e-2, top=3e-1)
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # ax.set_yscale("log", base=10)
        # plt.minorticks_off()
        # Set the exponent to be displayed only once at the top
        # ax.tick_params(axis="y", which="minor", labelleft=False)

        plt.xticks(fontsize=config['xtick_fontsize'])
        plt.yticks(fontsize=config['ytick_fontsize'])

        ax.set_xscale("log", base=10)

        # add some padding to the title
        f.suptitle(
            # (r"$\widehat{D}(\mathbb{P}_{|}, Q_{|})\,\,"
            #  r"\mathrm{between\,\, Protein\,\, MPNN\,\, and\,\,the \,\, CATH \,\, S60 \,\, distribution}$"),
            (r"$\mathrm{Comparing\,\,Protein MPNN\,\,and\,\,CATH\,\,S60\,\,proteins}$"),
            fontsize=config['title_fontsize'],
            # pad=30,
            wrap=True,
        )
        plt.tight_layout()



    whole_dataset_mmd_folder = Path("whole_dataset_plots")
    whole_dataset_mmd_folder.mkdir(exist_ok=True)

    if save:
        if filename_override is not None:
            filename = filename_override
        else:
            if line_sweep_quantity is not None:
                filename = (
                    f"./mmd_cgof_{sweep_quantity}_sweep_{line_sweep_quantity}_line_{'__'.join([f'{k}_{v}' for k, v in fixed_params.items()])}_{type_}.pdf"
                )
            else:
                filename = f"./mmd_cgof_{sweep_quantity}_sweep_{'__'.join([f'{k}_{v}' for k, v in fixed_params.items()])}_{type_}.pdf"

        print(f"Saving to {whole_dataset_mmd_folder / filename}")
        f.savefig(
            whole_dataset_mmd_folder / filename,
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=300,
        )

# parser = argparse.ArgumentParser()
# parser.add_argument("--type", type=str, default="cgof")  # cgof or calibration

if __name__ == "__main__":
    # args = parser.parse_args()
    # type_ = args.type


    # plot_sweep(
    #     df,
    #     sweep_quantity="temperature",
    #     # num_samples=5000 if type_ == "cgof" else 3000,
    #     num_samples=5000,
    #     type_=type_,
    # )

    # # dfs = df.loc[df["temperature"].isin([0.01, 0.03, 0.05, 0.1])]
    # # plot_sweep(dfs, sweep_quantity="num_samples", line_sweep_quantity="temperature")

    plt.ion()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    for (ax, type_) in zip((ax1, ax2), ["cgof", "calibration"]):
        if type_ == "cgof":
            df = pd.read_csv('../results/result_mmd_on_all_limited_redundancy_dataset.csv')
        elif type_ == "calibration":
            df = pd.read_csv('../results/result_skce_on_all_limited_redundancy_dataset.csv')
        else:
            raise ValueError(f"Unknown type {type_}")

        df = df.iloc[:, 2:]
        if type_ == "cgof":
            df = df.loc[df['torch_seed'] > 5]

        plot_sweep(
            df,
            sweep_quantity="temperature",
            # num_samples=5000 if type_ == "cgof" else 3000,
            num_samples=5000,
            type_=type_,
            ax=ax,
            f=f,
            filename_override=f"cgof_and_calibration.pdf",
            save=(ax is ax2),
        )
