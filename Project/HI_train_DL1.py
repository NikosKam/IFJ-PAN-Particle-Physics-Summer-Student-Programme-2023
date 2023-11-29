import sys
import h5py
import numpy as np
import pandas as pd
import json
import os
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import tensorflow as tf 
print(tf.__version__)
print('1: ', tf.config.list_physical_devices('GPU'))
print('2: ', tf.test.is_built_with_cuda)
print('3: ', tf.test.gpu_device_name())
print('4: ', tf.config.get_visible_devices())

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Input, add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping

from  datetime import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import socket
print (socket.gethostname())
print (os.getcwd())

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

import psutil
pid = os.getpid()
python_process = psutil.Process(pid)
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use at the beginning:', memoryUse)

lr_str = sys.argv[1]
batch_size_str = sys.argv[2]
lr = float(sys.argv[1])
batch_size = int(sys.argv[2])
number_of_epochs = int(sys.argv[3])

nubmer_of_initial_epoch=0

############# Picking method that calculates the loss
loss_method = 'mse'
#loss_method = 'categorical_crossentropy'
#loss_method = 'mean_squared_logarithmic_error'


#trainfile_name = 'training_even_dijets_2018_v0112_cent0_ter1.h5'
trainfile_name = 'events_even/training_even_jz2.h5'
h5f_train_1 = h5py.File(trainfile_name, 'r')
print("train 1: ",trainfile_name)
print(' ')
X_train = h5f_train_1['X_train'][:]
Y_train = h5f_train_1['Y_train'][:]
#X_train = h5f_train_1['X_train'][:595000]
#Y_train = h5f_train_1['Y_train'][:595000]

memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use after openning train file & fetching X_train Y_train 1:', memoryUse)

print(' ')
print ('X_train',type(X_train), 'len() : ', len(X_train))

dx_train = tf.data.Dataset.from_tensor_slices(X_train)
dy_train = tf.data.Dataset.from_tensor_slices(Y_train)

print(' ')
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use after creating dx/dy_train :', memoryUse)

print(' ')

#testfile_name = 'valt_odd_dijets_2018_v0112_cent0_ter2.h5'
testfile_name = 'events_odd/testing_odd_jz2.h5'
print('val: ',testfile_name)
print(' ')

h5f_test = h5py.File(testfile_name, 'r')
#2018 incl
#X_test = h5f_test['X_train'][:2250000] #In case Dom is lazy
#Y_test = h5f_test['Y_train'][:2250000] #In case Dom is lazy
X_test = h5f_test['X_test'][:2250000]
Y_test = h5f_test['Y_test'][:2250000]
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use after openning valt file', memoryUse)
print(' ')

dx_test = tf.data.Dataset.from_tensor_slices(X_test)
dy_test = tf.data.Dataset.from_tensor_slices(Y_test)
print ('X_val',type(dx_test), 'len(): ', len(dx_test))
print ('Y_val',type(dy_test), 'len(): ', len(dy_test))
print(' ')
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory after creating dx_test dy_test:', memoryUse)
print(' ')

# for def pt cuts batch_size=500, for figher cuts batch_size=1000
# for all v5* batch_size = 1000
#batch_size = 500
print(' ')
print('batch_size: ',batch_size)
print(' ')

steps = len(dy_train) / batch_size
print('training steps ',steps)
print(' ')
valid_steps = int(len(Y_test) / batch_size)
print('valid_steps: ',valid_steps)
print(' ')

#valid_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).repeat().batch(batch_size)
#iter = valid_dataset.make_one_shot_iterator()

train_dataset = tf.data.Dataset.zip(
        (dx_train, dy_train)).repeat().batch(batch_size)
valid_dataset = tf.data.Dataset.zip(
        (dx_test, dy_test)).repeat().batch(batch_size)
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory after zip dx/dy_train dx/dy_test:', memoryUse)
print(' ')

# Input layer
inputs = Input(shape=(X_train.shape[1],))
print('deb input ',(X_train.shape[1],) )

# number of nodes in the different hidden layers
#l_units = [72, 57, 60, 48, 36, 24, 12, 6]
l_units = [72, 60, 57, 48, 36, 24, 12, 6]
x = inputs
# loop to initialise the hidden layers
for unit in l_units:
    x = Dense(units=unit, activation="linear", kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#   x = Dropout(0.2)(x)
# output layer, using softmax which will return a probability for each jet to be either light, c- or b-jet
predictions = Dense(units=3, activation='softmax',
                    kernel_initializer='glorot_uniform')(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()

#lr=0.01
print(' ')
print('learning rate: ',lr)
print(' ')
model_optimizer = Adam(learning_rate=lr)
#model_optimizer = Adam(learning_rate=0.005,amsgrad=True)
model.compile(   loss= loss_method,
    optimizer=model_optimizer,
    metrics=['accuracy']
)

def GetRejection(y_pred, y_true):
#    Calculates the c and light rejection for 77% WP and 0.018 c-fraction.
    b_index, c_index, u_index = 2, 1, 0
    cfrac = 0.08
    target_beff = 0.77
    #print('deb 1 y_true ',y_true[:10])
    y_true = np.argmax(y_true, axis=1)
    #print('deb 2 y_true ',y_true[:20])
    #print('deb 2 u_pred',y_pred[:20])
    b_jets = y_pred[y_true == b_index]
    c_jets = y_pred[y_true == c_index]
    u_jets = y_pred[y_true == u_index]
    bscores = np.log(b_jets[:, b_index] / (cfrac * b_jets[:, c_index] +
                                           (1 - cfrac) * b_jets[:, u_index]))
    cutvalue = np.percentile(bscores, 100.0 * (1.0 - target_beff))

    c_eff = len(c_jets[np.log(c_jets[:, b_index] / (cfrac * c_jets[:, c_index]
                                                    + (1 - cfrac) *
                                                    c_jets[:, u_index])) >
                       cutvalue]) / float(len(c_jets))
    u_eff = len(u_jets[np.log(u_jets[:, b_index] / (cfrac *
                                                    u_jets[:, c_index] +
                                                    (1 - cfrac) *
                                                    u_jets[:, u_index])) >
                       cutvalue]) / float(len(u_jets))

    if c_eff == 0 or u_eff == 0:
        return -1, -1
    return 1. / c_eff, 1. / u_eff#, b_jets[:, b_index], c_jets[:, c_index], u_jets[:, u_index]

class MyCallback(Callback):
#Custom callback function calculating per epoch light and c-rejection and saves the model of each epoch.
#    def __init__():
    def __init__(self, X_valid=None, Y_valid=None, validation_data=None,
                 model_name='test', store_all=False, validation_steps = 0):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.validation_data = validation_data
        self.result = []
        self.model_name = model_name
        os.system("mkdir -p %s" % self.model_name)
        self.dict_list = []
        self.store_all = store_all
        self.validation_steps=validation_steps
    def on_epoch_end(self, epoch, logs=None):
        #memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        #print('on_epoch_end - deb :', memoryUse)

        if self.store_all:
            self.model.save('%s/model_epoch%i.h5' % (self.model_name, epoch))
        y_pred = self.model.predict(x=self.validation_data,steps=self.validation_steps)

        c_rej, u_rej = GetRejection(y_pred, self.Y_valid)

        print('on_epoch_end: len(y_pred): ',len(y_pred), ' len(Y_valid): ', len(self.Y_valid),'u_rej',u_rej)
        memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        print('memory use:', memoryUse)

        dict_epoch = {
            "epoch": epoch,
            "loss": logs['loss'],
            "accuracy": logs['accuracy'],
            "val_loss": logs['val_loss'],
            "val_accuracy": logs['val_accuracy'],
            "c_rej": c_rej,
            "u_rej": u_rej
        }

        self.dict_list.append(dict_epoch)
        with open('%s/DictFile_%s_%s_%s.json' % (self.model_name,str(lr_str),str(batch_size_str),str(loss_method)), 'w') as outfile:
            json.dump(str(self.dict_list), outfile, indent=4)

        memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        print('memory use on_epoch_end end: ', memoryUse)
        print(' ')

#early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=20, mode='auto')
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
#                              patience=5, min_lr=0.00000001,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.00001)

my_callback = MyCallback(
                         X_valid=X_test,
                         Y_valid=Y_test,
                         validation_data=valid_dataset,
                         model_name="DL1_test",
                         store_all=False, #flag to store model of each epoch
                         validation_steps=valid_steps
                        )

#callbacks = [reduce_lr, my_callback,early_stop]
callbacks = [reduce_lr, my_callback]

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Time before model.fit =", current_time)

model.fit(x=train_dataset,
          validation_data=valid_dataset,
          epochs=number_of_epochs, # typically ~130 are necessary to converge
          initial_epoch=nubmer_of_initial_epoch,
          steps_per_epoch=steps,
          validation_steps=len(Y_test) / batch_size,
          callbacks=callbacks,
          verbose=2
         )

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Time after model.fit =", current_time)

with open('DL1_test/DictFile_%s_%s_%s.json'%(str(lr_str),str(batch_size_str),str(loss_method))) as f:
    ata = json.load(f)
#print(ata)
#ta=ata.strip("[").strip("]")
#print(ta)
ata = ata.replace("\'", "\"")
a_json = json.loads(ata)
#print(a_json,type(a_json))
a_json

#dat = pd.read_json(a_json)
df_results = pd.read_json(ata)
print(df_results)

plt.plot(df_results['epoch'],df_results['accuracy'], label='training accuracy')
plt.plot(df_results['epoch'],df_results['val_accuracy'], label='validation accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.savefig('pdf_files/acc_vs_epochs_incl_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.pdf')
plt.close()

plt.plot(df_results['epoch'],df_results['loss'], label='training loss')
plt.plot(df_results['epoch'],df_results['val_loss'], label='validation loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig('pdf_files/loss_vs_epochs_incl_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.pdf')
plt.close()

plt.plot(df_results['epoch'],df_results['u_rej'], label='light flavour jet rejection')
plt.plot(df_results['epoch'],df_results['c_rej'], label='c- jet rejection')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('light flavour jet rejection')
plt.savefig('pdf_files/rej_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.pdf')
plt.close()

# get the architecture as a json string
arch = model.to_json()
# save the architecture string to a file somehow, the below will work
with open('net/architecture_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.json', 'w') as arch_file:
    arch_file.write(arch)
# Save a model you have trained
model.save('net/trained_model_lr' + str(lr_str) + '_bs' + str(batch_size_str)+'_'+str(loss_method)+'.keras')
# now save the weights as an HDF5 file
model.save_weights('net/weights_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.h5')

valaccuracy_txt = 'txt_test/valaccuracy_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
accuracy_txt = 'txt_test/accuracy_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
np.savetxt(valaccuracy_txt, list(zip(df_results["epoch"],df_results['val_accuracy'])))
np.savetxt(accuracy_txt, list(zip(df_results["epoch"],df_results['accuracy'])))

valloss_txt = 'txt_test/valloss_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
loss_txt = 'txt_test/loss_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
np.savetxt(valloss_txt, list(zip(df_results["epoch"],df_results['val_loss'])))
np.savetxt(loss_txt, list(zip(df_results["epoch"],df_results['loss'])))

lrej_txt = 'txt_test/lrej_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
crej_txt = 'txt_test/crej_vs_epochs_lr'+str(lr_str)+'_bs'+str(batch_size_str)+'_'+str(loss_method)+'.txt'
np.savetxt(lrej_txt, list(zip(df_results["epoch"],df_results['u_rej'])))
np.savetxt(crej_txt, list(zip(df_results["epoch"],df_results['c_rej'])))

memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use at the end: ', memoryUse)
