import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import re

sys.path.append('../')

from Architecture import surrogate_net
from Utils.scalers import MinMaxScaler
from Utils import NAMES

def train_for_region(varname, filename, region_number, region_name):

    df = pd.read_csv('../Datasets/POSTPRO_RESULTS_10MW_filtered.csv')

    if 'del' in varname:
        groups = re.search(r"m(\d+)_.*_\[(.*)\]", varname)
        m = groups.group(1)
        unit = groups.group(2)
    else:
        groups = re.search(r".*_\[(.*)\]", varname)
        m = None
        unit = groups.group(1)

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

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    net = surrogate_net()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    losses = []
    val_losses = []
    best_val_loss = 1e10

    for epoch in range(3000):
        
        optimizer.zero_grad()
        output = net(torch.Tensor(X_train))
        loss = criterion(output, torch.Tensor(Y_train).view(-1, 1))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            output_val = net(torch.Tensor(X_val))
            loss_val = criterion(output_val, torch.Tensor(Y_val).view(-1, 1))
            val_losses.append(loss_val.item())
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                net_info = {
                    'state_dict': net.state_dict(),
                    'Xmin': input_scaler.min,
                    'Xmax': input_scaler.max,
                    'Ymin': output_scaler.min,
                    'Ymax': output_scaler.max,
                    'n_samples': len(X_train),
                    'n_inputs': len(X_train[0]),
                    'input_channel_names': ['Wind Speed', 'Turbulence Intensity', 'Shear Exponent', 'Yaw Angle'],
                    'input_channel_units': ['m/s', '-', '-', 'deg'],
                    'n_outputs': 1,
                    'output_channel_names': [NAMES[varname]],
                    'output_channel_units': [unit],
                    'wohler_exponent': m
                }
                torch.save(net_info, f"./models/{region_name}/{filename}_best.pth")
                print(f'Epoch {epoch} - Loss: {loss.item()} - Val Loss: {loss_val.item()}')

    # r2 score
    Y_pred = net(torch.Tensor(X))
    Y_pred = Y_pred.detach().numpy().flatten()
    residuals = Y - Y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1 - (ss_res/ss_tot)
    print(f'[{region_name}] R2 score: {r2}')

    # Map Y and Y_pred back to original scale
    Y_t = output_scaler.inverse_transform(Y)
    Y_pred_t = output_scaler.inverse_transform(Y_pred)
    X_t = input_scaler.inverse_transform(X)

    fig1, ax1 = plt.subplots()

    ax1.set_title(f'{region_name} - {NAMES[varname]}')
    ax1.plot(X_t[:, 0], Y_pred_t, 'o')
    ax1.plot(X_t[:, 0], Y_t, 'x')

    # Calculate rmspe
    rmspe = np.sqrt(np.mean(np.square(((Y_pred_t - Y_t) / Y_t)), axis=0))
    print(f'[{region_name}] RMSPE: {rmspe}')

    # Calculate rmse
    rmse = np.sqrt(np.mean(np.square(Y_pred_t - Y_t), axis=0))
    print(f'[{region_name}] RMSE: {rmse}')

    # Calculate mae
    mae = np.mean(np.abs(Y_pred_t - Y_t), axis=0)
    print(f'[{region_name}] MAE: {mae}')

    fig2, ax2 = plt.subplots()

    ax2.set_title(f'{region_name} - {NAMES[varname]}')
    ax2.plot(losses)
    ax2.plot(val_losses)
    plt.show()


if __name__ == '__main__':

    TRAINABLE_VARIABLES = [('power_[kW]', 'POWER'), ('ct_[-]', 'CT'), ('m10_del_blew_avg_[kN-m]', 'DEL_BLEW'), ('m10_del_blfw_avg_[kN-m]', 'DEL_BLFW'), ('m7_del_ttyaw_[kN-m]', 'DEL_TTYAW'), ('m4_del_tbss_[kN-m]', 'DEL_TBSS'), ('m4_del_tbfa_[kN-m]', 'DEL_TBFA')]
    TRAINABLE_REGIONS = [('low', 0), ('mid', 1), ('nonop', 2)]

    for varname, filename in TRAINABLE_VARIABLES:
        for region_name, idx in TRAINABLE_REGIONS:
            if not os.path.exists(f'./models/{region_name}'):
                os.makedirs(f'./models/{region_name}')
            train_for_region(varname, filename, idx, region_name)
