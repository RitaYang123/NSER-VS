# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:59:35 2024

@author: Rui Yang
"""

import sys

from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from E_SAR_LassoNet import LassoNetSARExpectileRegressor
import os
import math
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn.functional as F
import Gendata

#%%
####################Linear_sparse######################
device = torch.device("cpu")

N = 500
d = 10
rho=0.3
expectile=0.25
I_tensor = torch.tensor(np.eye(N), dtype=torch.float32)  # Convert to PyTorch tensor


trainlist, validationlist, testlist = Gendata.Generate(N=N, d=d, rho=rho, matrix='er0.5',
                                                                   model='linear_sparse', err='K')

# train
W_tensor = torch.FloatTensor(trainlist[0]).to(device)
mask = torch.FloatTensor(trainlist[1]).to(device)
x_tensor = torch.FloatTensor(trainlist[2]).to(device)
y_tensor = torch.FloatTensor(trainlist[3]).to(device)
err = torch.FloatTensor(trainlist[4]).to(device)
# validation
W_tensor_validation = torch.FloatTensor(validationlist[0]).to(device)
x_tensor_validation = torch.FloatTensor(validationlist[2]).to(device)
y_tensor_validation = torch.FloatTensor(validationlist[3]).to(device)
err_validation = torch.FloatTensor(validationlist[4]).to(device)
# test
W_tensor_test = torch.FloatTensor(testlist[0]).to(device)
x_tensor_test = torch.FloatTensor(testlist[2]).to(device)
y_tensor_test = torch.FloatTensor(testlist[3]).to(device)
err_test = torch.FloatTensor(testlist[4]).to(device)


model = LassoNetSARExpectileRegressor(
    hidden_dims=(10, 10,),
    M=26,
    path_multiplier=1.2,
    verbose=True,
    expectile=expectile,
    torch_seed=1,
    lambda_start=0.01,
    device=device,
    )
path = model.path(x_tensor, y_tensor, W_tensor)


 
