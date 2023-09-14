# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: train and evaluate EGPDM in NASA dataset using transfer learning
# @license: (C) Copyright 2023, Ice Lab Limited.

import numpy as np
import pandas as pd
import time
from model.egpdm_v1 import EGPDM
from utils import interpolate_data, rmse, normalize_data, denormalize_data, plot_battery_degradation


""" hyper-parameters """
Q = 3           # latent dim
epochs = 2      # optimization steps (max epochs)
lr = 0.01       # learning rate
ratio = 0.5    # portion of the test sequence given


""" load NASA dataset """
# get 4D data of (B0005, B0006 & B0007) from the same group
# each have shape (N=168, D=4)
b5, _ = normalize_data(np.array(pd.read_csv('data/original/NASA_B0005.csv', header=None)))
b6, _ = normalize_data(np.array(pd.read_csv('data/original/NASA_B0006.csv', header=None)))
b7, test_scaler = normalize_data(np.array(pd.read_csv('data/original/NASA_B0007.csv', header=None)))

# we use B0005, B0006 and part of B0007 as training points
# the rest of B0007 as test points
b7_tr = b7[:int(b7.shape[0] * ratio),:]
# interpolate train part of B0007 to align the length (timestep)
b7_tr_align = interpolate_data(b7_tr, target_len=b7.shape[0])
Y_data = [b5, b6, b7_tr_align]

""" init EGPDM """
D = Y_data[0].shape[1]
model = EGPDM(D=D, Q=Q, dyn_target='full')

# add pre-train data
for i in Y_data:
    model.add_data(i)

# initialize X init by PCA
model.init_X()

""" train EGPDM """
start_time = time.time()
loss = model.train_lbfgs(num_opt_steps=epochs, lr=lr, balance=1)
end_time = time.time()
train_time = end_time - start_time
print("\nTotal Training Time: "+str(train_time)+" s")


""" plot results """
model.eval()
X_list = model.get_latent_sequences()
Y_list = model.observations_list
X = X_list[-1]
Y = Y_list[-1]
N = Y.shape[0]  # timestep
forward_steps = N - b7_tr.shape[0]  # how many steps to inference
# choose the end of the seq to rollout
_, Ypred, Ystd = model(num_steps=forward_steps, num_sample=100, X0=X[-1,:], flg_noise=True)

# denormalize
Ypred = denormalize_data(Ypred, test_scaler)
b7 = denormalize_data(b7, test_scaler)
b7_tr = b7[:int(b7.shape[0] * ratio),:]
b7_te = b7[int(b7.shape[0] * ratio):,:]

# plot
plot_battery_degradation(b7, ratio, Ypred, Ystd, forward_steps)

# calculate results
print('\n ### RESULT ###')
print('normalized rmse: ' + str(rmse(b7_te[:,0],Ypred[:,0])))