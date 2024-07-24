import pandas as pd
import numpy as np
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import config

plt.style.use("figures.mplstyle")

name_map = {
    "upsample_to": r"Num. Samples",
    "delta_t": r"Temperature Difference",
    # "val": r"$\\widehat{D}(\\mathbb P_{|}, Q_{|})$",
    # "biased_val": r"$\\widehat{D}(\\mathbb P_{|}, Q_{|})$ (biased)",
    "val": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$",
    "biased_val": r"$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$ (biased)",
    "result": r"Rejection rate",
}

CATH_name_key = {
    '2.40.70.10' : 'Acid Proteases',
    '3.10.20.90' : 'PI 3-kinase',
    '1.10.20.10' : 'Histone',
    '2.30.30.100': 'Integrase',
    '3.30.70.330': 'RRM domain',
    '2.40.50.40': '2.40.50.40',
    '2.30.30.40': 'SH3 domains',
    '2.60.40.10': 'Immunoglobulins',
    '1.10.10.10': 'WHL DNA binding',
    '1.10.10.60': 'Homeodomain-like',
    '1.10.238.10': 'EF-hand'
}


def barplot(df):
    mean_val = df[['val', 'Superfamily', 'temperature']].groupby(['Superfamily', 'temperature'])['val'].mean().unstack()
    mean_val = mean_val.sort_values(by=mean_val.columns[0], ascending=False)


    std_val = df[['val', 'Superfamily', 'temperature']].groupby(['Superfamily', 'temperature'])['val'].std().unstack()
    std_val = std_val.loc[mean_val.index, :]

    f, ax = plt.subplots(figsize=(8, 4))
    # with mpl.rc_context(fname="figures.mplstyle"):

    import numpy as np
    xs = np.arange(len(mean_val.index))


    mean_vals_t0 = mean_val.iloc[:, 0]
    mean_vals_t1 = mean_val.iloc[:, 1]

    std_vals_t0 = std_val.iloc[:, 0]
    std_vals_t1 = std_val.iloc[:, 1]

    xs_t0 = xs - 0.2
    xs_t1 = xs + 0.2
    # ax.bar(mean_val.index, mean_val['val'])
    ax.bar(
        xs_t0, mean_vals_t0, width=0.4, label=mean_val.columns[0],
        # errorbars
        yerr=std_vals_t0,
    )
    ax.bar(
        xs_t1,
        mean_vals_t1,
        width=0.4,
        label=mean_val.columns[1],
        # errorbars
        yerr=std_vals_t1,
    )



    ax.set_xticks(xs)
    latex_ticklabels = [v for v in mean_val.index]

    ax.set_xticklabels(
        latex_ticklabels,
        rotation=45,
        ha='right',
        fontsize=config['xtick_fontsize']
    )

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(
        loc='center right', bbox_to_anchor=(1.30, 0.5),
        title=r"$T$",
        title_fontsize=config["legend_title_fontsize"],
    )

    ax.set_ylabel(name_map['val'], fontsize=config['ylabel_fontsize'])
    ax.set_xlabel('Superfamily', fontsize=config['xlabel_fontsize'])

    ax.set_title(
        r'$\widehat{\mathrm{ACMMD}}(\mathbb{P}_{|}, Q_{|})$ on various CATH superfamilies',
        fontsize=config['title_fontsize'],
        pad=20,
    )

    from pathlib import Path
    folder = Path('superfamily_plots')
    folder.mkdir(exist_ok=True)
    f.savefig(folder / 'superfamily_T_bar.pdf', bbox_inches='tight')

if __name__ == '__main__':
    
    plt.ion()

    fnm = '../results/result_mmd_on_true_data.csv'
    df = pd.read_csv(fnm, index_col=0)

    cath_cols = ['Class Number', 'Architecture Number', 'Topology Number',
            'Superfamily Number']

    #add in nested annotations
    for i in range(4):
        S = df[cath_cols[0]].astype(str)
        nm = 'C'
        if i > 0:
            for j in range(1,i+1):
                S += '.' + df[cath_cols[j]].astype(str)
                nm += 'CATH'[j]
        df[nm] = S

    df['Superfamily'] = [CATH_name_key[h] for h in df['CATH']]

    barplot(df)
