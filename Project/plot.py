# importing necessary packages
import sys
import h5py
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pandas as pd
import json
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
#from keras.layers.normalization import layer_normalization
#from keras.layers import LayerNormalization

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Input, add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping

from eplottingFunctions import plotBin, plotBinReal

model_name = "net/trained_model_lr0.1_bs2000_mean_squared_logarithmic_error.keras"
#model_name = "net/trained_model_lr0.05_bs2000_mean_squared_logarithmic_error.keras"
#model_name = "net/trained_model_lr0.1_bs1000_mse.keras"
#model_name = "net/trained_model_lr0.05_bs1000_mse.keras"
#model_name = "net/trained_model_lr0.1_bs2000_mse.keras"
#model_name = "net/trained_model_lr0.05_bs2000_mse.keras"
#model_name = "net/trained_model_lr0.05_bs5000_mse.keras"
#model_name = "net/trained_model_lr0.05_bs2000_categorical_crossentropy.keras"


loaded_model = load_model(model_name)

loaded_model.summary()

#file = 'testing_dijets_2018_v0112_cent0_ter2_2.h5'
file = 'events_odd/testing_odd_jz2.h5'
#file = 'real_data_testing_wow_jz2.h5'
h5f_test_1 = h5py.File(file, 'r')
print("test 1: ",file)
print(' ')
X_test = h5f_test_1['X_test'][:]
#X_test = X_test.astype(np.float32)
Y_test = h5f_test_1['Y_test'][:]

print('Result name: ', model_name[18:])

plotBin(loaded_model, X_test, Y_test, returnDisc=True, fc=0.08, fig_name='dl1_discriminant.pdf', jz='incl', ver='77', file = '_sup', name = model_name[18:])
#plotBinReal(loaded_model, X_test, returnDisc=True, fc=0.08, fig_name='dl1_discriminant_Data.pdf', jz='incl', ver='77', file = '_sup', name = model_name[18:])