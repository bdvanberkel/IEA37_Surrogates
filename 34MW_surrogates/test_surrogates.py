import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

sys.path.append('../')

from Architecture import surrogate_net
from Utils.scalers import MinMaxScaler
from Utils import NAMES, SYMBOLS

def test_for_variable(varname, filename, regions):

    print(f'Testing for {varname}...')

    if 'del' in varname:
        groups = re.search(r"m(\d+)_.*_\[(.*)\]", varname)
        m = groups.group(1)
        unit = groups.group(2)
    else:
        groups = re.search(r".*_\[(.*)\]", varname)
        m = None
        unit = groups.group(1)

    all_X = None
    all_Y = None
    all_pred = None

    all_X_t = None
    all_Y_t = None
    all_pred_t = None

    def test_for_region(varname, filename, region_number, region_name):

        df = pd.read_csv('../Datasets/POSTPRO_RESULTS_34MW_filtered.csv')

        if region_number == 0:
            df = df[(df['ws_[m/s]'] >= 0) & (df['ws_[m/s]'] < 4) & ~df['parked_[bool]']]
        elif region_number == 1:
            df = df[(df['ws_[m/s]'] >= 4) & (df['ws_[m/s]'] <= 25) & ~df['parked_[bool]']]
        elif region_number == 2:
            df = df[(df['ws_[m/s]'] > 25) & (df['ws_[m/s]'] <= 30) | df['parked_[bool]']]

        X = df[['ws_[m/s]', 'ti_[-]', 'alpha_[-]', 'yaw_[deg]']]
        Y = df[varname]

        input_scaler = MinMaxScaler(X.values)
        output_scaler = MinMaxScaler(Y.values)

        # min-max normalization
        X = input_scaler.transform(X.values)
        Y = output_scaler.transform(Y.values)

        net = surrogate_net()
        model_data = torch.load(f"./models/{region_name}/{filename}_best.pth")
        net.load_state_dict(model_data['state_dict'])
        net.eval()

        # r2 score
        Y_pred = net(torch.Tensor(X))
        Y_pred = Y_pred.detach().numpy().flatten()

        # Map Y and Y_pred back to original scale
        Y_t = output_scaler.inverse_transform(Y)
        Y_pred_t = output_scaler.inverse_transform(Y_pred)
        X_t = input_scaler.inverse_transform(X)

        return X, Y, Y_pred, X_t, Y_t, Y_pred_t

    for region_name, idx in regions:
        X, Y, Y_pred, X_t, Y_t, Y_pred_t = test_for_region(varname, filename, idx, region_name)

        if all_X is None:
            all_X = X
            all_Y = Y
            all_pred = Y_pred
            all_X_t = X_t
            all_Y_t = Y_t
            all_pred_t = Y_pred_t
        else:
            all_X = np.concatenate([all_X, X])
            all_Y = np.concatenate([all_Y, Y])
            all_pred = np.concatenate([all_pred, Y_pred])
            all_X_t = np.concatenate([all_X_t, X_t])
            all_Y_t = np.concatenate([all_Y_t, Y_t])
            all_pred_t = np.concatenate([all_pred_t, Y_pred_t])

    all_pred = all_pred.flatten()
    residuals = all_Y - all_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_Y - np.mean(all_Y))**2)
    r2 = 1 - (ss_res/ss_tot)
    print(f'[low+mid+high] R2 score: {r2}')

    # fig1, ax1 = plt.subplots()

    # ax1.scatter(all_X_t[:, 0], all_pred_t, marker='o', c=all_X_t[:, 3], s=2, cmap='viridis')
    # # ax1.plot(all_X_t[:, 0], all_Y_t, 'x')
    # ax1.set_xlabel('Wind Speed [m/s]')
    # ax1.set_ylabel(f'{NAMES[varname]} [{unit}]')
    # fig1.colorbar(ax1.scatter(all_X_t[:, 0], all_pred_t, marker='o', c=all_X_t[:, 3], s=2, cmap='viridis'), label='Yaw Angle [deg]')
    # plt.tight_layout()

    # Calculate rmspe
    rmspe = np.sqrt(np.mean(np.square(((all_pred_t - all_Y_t) / all_Y_t)), axis=0))
    print(f'[low+mid+high] RMSPE: {rmspe}')

    # Calculate rmse
    rmse = np.sqrt(np.mean(np.square(all_pred_t - all_Y_t), axis=0))
    print(f'[low+mid+high] RMSE: {rmse}')

    # Calculate mae
    mae = np.mean(np.abs(all_pred_t - all_Y_t), axis=0)
    print(f'[low+mid+high] MAE: {mae}')

    # Calculate mape
    mape = np.mean(np.abs((all_pred_t - all_Y_t) / all_Y_t), axis=0)
    print(f'[low+mid+high] MAPE: {mape}')

    # plt.show()

    return all_X_t, all_Y_t, all_pred_t, NAMES[varname], unit, r2, rmspe, rmse, mae, mape

if __name__ == '__main__':

    TESTABLE_VARIABLES = [
        ('power_[kW]', 'POWER', [('mid', 1)]), # Only mid region as we'll set the power to 0 for the other regions
        ('ct_[-]', 'CT', [('mid', 1)]), # Only mid region as we'll set the power to 0 for the other regions
        ('m10_del_blew_avg_[kN-m]', 'DEL_BLEW', [('low', 0), ('mid', 1), ('nonop', 2)]),
        ('m10_del_blfw_avg_[kN-m]', 'DEL_BLFW', [('low', 0), ('mid', 1), ('nonop', 2)]),
        ('m7_del_ttyaw_[kN-m]', 'DEL_TTYAW', [('low', 0), ('mid', 1), ('nonop', 2)]),
        ('m4_del_tbss_[kN-m]', 'DEL_TBSS', [('low', 0), ('mid', 1), ('nonop', 2)]),
        ('m4_del_tbfa_[kN-m]', 'DEL_TBFA', [('low', 0), ('mid', 1), ('nonop', 2)])
    ]

    fig, axs = plt.subplots(len(TESTABLE_VARIABLES), 4, figsize=(20, 4*len(TESTABLE_VARIABLES)))

    for i, (varname, filename, regions) in enumerate(TESTABLE_VARIABLES):
        all_X_t, all_Y_t, all_pred_t, name, unit, r2, rmspe, rmse, mae, mape = test_for_variable(varname, filename, regions)

        ax1 = axs[i, 0]
        ax2 = axs[i, 1]
        ax3 = axs[i, 2]
        ax4 = axs[i, 3]

        sc = ax1.scatter(all_X_t[:, 0], all_pred_t, marker='o', c=all_X_t[:, 3], s=2, cmap='viridis')
        # ax1.plot(all_X_t[:, 0], all_Y_t, 'x')
        ax1.set_xlabel('Wind Speed [m/s]')
        ax1.set_ylabel(SYMBOLS[varname])
        ax1.set_title(f'{SYMBOLS[varname]} vs {SYMBOLS["ws_[m/s]"]}')
        ax1.set_xlim([0, 30])
        ax1.set_ylim([0, max(all_pred_t)])
        fig.colorbar(sc, label='Yaw Angle [deg]')

        sc2 = ax2.scatter(all_X_t[:, 3], all_pred_t, marker='o', c=all_X_t[:, 0], s=2, cmap='cividis')
        # ax1.plot(all_X_t[:, 0], all_Y_t, 'x')
        ax2.set_xlabel('Yaw Angle [deg]')
        ax2.set_ylabel(SYMBOLS[varname])
        ax2.set_title(f'{SYMBOLS[varname]} vs {SYMBOLS["yaw_[deg]"]}')
        ax2.set_xlim([-30, 30])
        ax2.set_ylim([0, max(all_pred_t)])
        fig.colorbar(sc2, label='Wind Speed [m/s]')

        ax3.set_title(f"{SYMBOLS[varname]} - {r'$Y$'} vs {r'$Y_{pred}$'}")
        sc3 = ax3.scatter(all_Y_t, all_pred_t, marker='o', s=2, c=all_X_t[:, 0], cmap='cividis')
        ax3.text(0.70, 0.05, f"{r'$R^2$'}: {r2:.3f}", transform=ax3.transAxes)
        ax3.plot([0, max(all_Y_t)], [0, max(all_Y_t)], 'r--')
        ax3.set_xlabel(SYMBOLS[varname])
        ax3.set_ylabel(SYMBOLS[varname])
        ax3.set_xlim([0, max(all_Y_t)])
        ax3.set_ylim([0, max(all_Y_t)])
        fig.colorbar(sc3, label='Wind Speed [m/s]')

        residuals = all_Y_t - all_pred_t
        ax4.hist(residuals, bins=100, color='cornflowerblue', edgecolor='cornflowerblue', density=True)
        ax4.set_title(f"Residuals for {SYMBOLS[varname]}")
        ax4.set_xlabel("Residuals")
        ax4.set_ylabel("Density")
    
    plt.tight_layout()
    plt.savefig('evaluation.png')