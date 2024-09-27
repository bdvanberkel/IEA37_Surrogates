import pandas as pd
import matplotlib.pyplot as plt

import sys
import re

sys.path.append('../')

from Utils import NAMES, SYMBOLS

def plot_variable_versus_ws(df, varnames):

    n = len(varnames)
    fig, axs = plt.subplots(n, 2, figsize=(10, 4*n))

    for i, varname in enumerate(varnames):

        if 'del' in varname:
            groups = re.search(r"m(\d+)_.*_\[(.*)\]", varname)
            m = groups.group(1)
            unit = groups.group(2)
        else:
            groups = re.search(r".*_\[(.*)\]", varname)
            m = None
            unit = groups.group(1)

        y_range = None

        for parked in [False, True]:

            ax = axs[i, int(parked)]

            df_i = df[df['parked_[bool]'] == parked]

            if varname in ['power_[kW]', 'ct_[-]']:
                df_i = df_i[(df_i['ws_[m/s]'] <= 25) & (df_i['ws_[m/s]'] >= 4)]

            if not parked:
                y_range = [df_i[varname].min(), df_i[varname].max()]
                # print(f'{varname} range: {y_range}')

                if varname in ['power_[kW]', 'ct_[-]']:
                    y_range[0] = 0

            parked_text = 'Parked' if parked else 'Operating'

            sc = ax.scatter(df_i['ws_[m/s]'], df_i[varname], marker='o', c=df_i['yaw_[deg]'], s=2, cmap='viridis')
            ax.set_xlabel(f'Wind Speed {SYMBOLS["ws_[m/s]"]}')
            ax.set_ylabel(SYMBOLS[varname])
            fig.colorbar(sc, label=f"Yaw Angle {SYMBOLS['yaw_[deg]']}")
            ax.set_title(f'{SYMBOLS[varname]} vs {SYMBOLS["ws_[m/s]"]}, {parked_text}')
            ax.set_ylim(y_range)
            ax.set_xlim([0, 30])

    plt.tight_layout()
    # plt.show()
    plt.savefig('DEL_vs_ws.png')

def plot_variable_versus_yaw(df, varnames):

    n = len(varnames)
    fig, axs = plt.subplots(n, 2, figsize=(10, 4*n))

    for i, varname in enumerate(varnames):

        if 'del' in varname:
            groups = re.search(r"m(\d+)_.*_\[(.*)\]", varname)
            m = groups.group(1)
            unit = groups.group(2)
        else:
            groups = re.search(r".*_\[(.*)\]", varname)
            m = None
            unit = groups.group(1)

        y_range = None

        for parked in [False, True]:

            ax = axs[i, int(parked)]

            df_i = df[df['parked_[bool]'] == parked]

            if varname in ['power_[kW]', 'ct_[-]']:
                df_i = df_i[(df_i['ws_[m/s]'] <= 25) & (df_i['ws_[m/s]'] >= 4)]

            if not parked:
                y_range = [df_i[varname].min(), df_i[varname].max()]
                # print(f'{varname} range: {y_range}')

                if varname in ['power_[kW]', 'ct_[-]']:
                    y_range[0] = 0

            parked_text = 'Parked' if parked else 'Operating'

            sc = ax.scatter(df_i['yaw_[deg]'], df_i[varname], marker='o', c=df_i['ws_[m/s]'], s=2, cmap='cividis')
            ax.set_xlabel(f'Yaw Angle {SYMBOLS["yaw_[deg]"]}')
            ax.set_ylabel(SYMBOLS[varname])
            fig.colorbar(sc, label=f"Wind Speed {SYMBOLS['ws_[m/s]']}")
            ax.set_title(f'{SYMBOLS[varname]} vs {SYMBOLS["yaw_[deg]"]}, {parked_text}')
            ax.set_ylim(y_range)
            ax.set_xlim([-30, 30])

    plt.tight_layout()
    # plt.show()
    plt.savefig('DEL_vs_yaw.png')

if __name__ == '__main__':

    df = pd.read_csv('../Datasets/POSTPRO_RESULTS_34MW_filtered.csv')

    PLOTTABLE_VARIABLES = [
        'power_[kW]',
        'ct_[-]',
        'm10_del_blew_avg_[kN-m]',
        'm10_del_blfw_avg_[kN-m]',
        'm7_del_ttyaw_[kN-m]',
        'm4_del_tbss_[kN-m]',
        'm4_del_tbfa_[kN-m]'
    ]

    plot_variable_versus_ws(df, PLOTTABLE_VARIABLES)
    plot_variable_versus_yaw(df, PLOTTABLE_VARIABLES)