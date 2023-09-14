# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: utils for projects
# @license: (C) Copyright 2023, AMMLE Group Limited.

import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def interpolate_data(data, target_len=168):
    N = data.shape[0]
    D = data.shape[1]
    new_x = np.linspace(0, N-1, target_len)
    new_data = np.zeros((target_len, D))

    for i in range(D):
        old_x = np.arange(N)
        old_y = data[:, i]
        interpolator = interpolate.interp1d(old_x, old_y, kind='linear')
        new_data[:, i] = interpolator(new_x)

    return new_data

def rmse(yte, ypred):
    return np.sqrt(np.mean(np.square(yte - ypred)))

def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def denormalize_data(data, scaler):
    return scaler.inverse_transform(data)

def plot_battery_degradation(Y, ratio, Ypred, Ystd, steps):
    N = Y.shape[0]
    Ytr = Y[:int(N * ratio),:]
    low_bound = Ypred[:, 0].min()
    print(low_bound)
    cycles = [i for i in range(N)]
    f, ax = plt.subplots()
    ax.plot(cycles, Y[:, 0], '-', color='#4c4c4c', linewidth=2, label="True capacity")
    ax.plot(cycles[:Ytr.shape[0]], Ytr[:, 0], 'k+', mew=2, ms=6, label="Observations")
    ax.plot(cycles[Ytr.shape[0]:Ytr.shape[0] + steps], Ypred[:, 0], "#d85218", label="Prediction",
            linewidth=3)
    ax.fill_between(cycles[Ytr.shape[0]:Ytr.shape[0] + steps], Ypred[:, 0] + np.sqrt(Ystd[:, 0]),
                    Ypred[:, 0] - np.sqrt(Ystd[:, 0]), color="#f5d8cb", alpha=0.8, label="Confidence")
    ax.vlines(x=Ytr.shape[0], ymin=-0.1, ymax=1.1, color="#d85218", linestyles="dashed")
    ax.set_xlim([0, N - 1])
    ax.set_ylim([low_bound - 0.1, 1.1])
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel('Cycles', fontsize=14)
    ax.set_ylabel('SOH', fontsize=14)
    ax.tick_params(width=2, labelsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(40))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.legend(frameon=False, loc='lower left')
    plt.show()
