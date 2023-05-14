# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: utils for projects
# @license: (C) Copyright 2023, AMMLE Group Limited.

import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

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
    return scaler.fit_transform(data)

def denormalize_data(data):
    return scaler.inverse_transform(data)