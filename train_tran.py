# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: train and evaluate GPDM in NASA dataset using transfer learning
# @license: (C) Copyright 2023, AMMLE Group Limited.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from model.gpdm import GPDM
from utils import interpolate_data, rmse


""" hyper-parameters """
Q = 2           # latent dim
epochs = 4      # optimization steps (max epochs)
lr = 0.01       # learning rate
ratio = 0.5     # portion of the test sequence given


""" load NASA dataset """
# get 4D data of (B0005, B0006 & B0007) from the same group
# each have shape (N=168, D=4)
b5 = np.array(pd.read_csv('data/normalized/NASA_B0005.csv', header=None))
b6 = np.array(pd.read_csv('data/normalized/NASA_B0006.csv', header=None))
b7 = np.array(pd.read_csv('data/normalized/NASA_B0007.csv', header=None))

# we use B0005, B0006 and part of B0007 as training points
# the rest of B0007 as test points
b7_tr = b7[:int(b7.shape[0] * ratio),:]
b7_te = b7[int(b7.shape[0] * ratio):,:]
# interpolate train part of B0007 to align the length (timestep)
b7_tr_align = interpolate_data(b7_tr, target_len=b7.shape[0])
Y_data = [b5, b6, b7_tr_align]

""" init GPDM """
D = Y_data[0].shape[1]
dyn_target = 'full'  # 'full' or 'delta'
model = GPDM(D=D, Q=Q, dyn_target=dyn_target)

# add pre-train data
for i in Y_data:
    model.add_data(i)

# initialize X init by PCA
model.init_X()

""" train GPDM """
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
# plot
cycles = [i for i in range(N)]
f, ax = plt.subplots()
ax.plot(cycles, b7[:,0], '-', color='#4c4c4c', linewidth=2, label="True capacity")
ax.plot(cycles[:b7_tr.shape[0]], b7_tr[:,0], 'k+', mew=2, ms=6, label="Observations")
ax.plot(cycles[b7_tr.shape[0]:b7_tr.shape[0]+forward_steps], Ypred[:, 0], "#d85218", label="Prediction", linewidth=3)
ax.fill_between(cycles[b7_tr.shape[0]:b7_tr.shape[0]+forward_steps], Ypred[:, 0] + np.sqrt(Ystd[:, 0]),
                 Ypred[:, 0] - np.sqrt(Ystd[:, 0]), color="#f5d8cb", alpha=0.8, label="Confidence")
ax.vlines(x=b7_tr.shape[0], ymin=-0.1, ymax=1.1, color="#d85218", linestyles="dashed")
ax.set_xlim([0, N-1])
ax.set_ylim([-0.1, 1.1])
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

# calculate results
print('\n ### RESULT ###')
print('normalized rmse: ' + str(rmse(b7_te[:,0],Ypred[:,0])))
print('Note that normalized rmse is greater than rmse of original data.')