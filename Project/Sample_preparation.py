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

even_or_odd = 'odd'
jz_slice = '2'

def DownSampling(bjets, cjets, ujets):
    # pT in MeV
    pt_bins = np.linspace(0, 800000, 100)
    eta_bins = np.linspace(-2.5, 2.5, 10)

    histvals_b, _, _ = np.histogram2d(bjets['abs_eta_uncalib'], bjets['pt_uncalib'],
                                [eta_bins, pt_bins])
    histvals_c, _, _ = np.histogram2d(cjets['abs_eta_uncalib'], cjets['pt_uncalib'],
                                [eta_bins, pt_bins])
    histvals_u, _, _ = np.histogram2d(ujets['abs_eta_uncalib'], ujets['pt_uncalib'],
                                [eta_bins, pt_bins])
    b_locations_pt = np.digitize(bjets['pt_uncalib'], pt_bins) - 1
    b_locations_eta = np.digitize(bjets['abs_eta_uncalib'], eta_bins) - 1
    b_locations = zip(b_locations_pt, b_locations_eta)
    b_locations = list(b_locations)

    c_locations_pt = np.digitize(cjets['pt_uncalib'], pt_bins) - 1
    c_locations_eta = np.digitize(cjets['abs_eta_uncalib'], eta_bins) - 1
    c_locations = zip(c_locations_pt, c_locations_eta)
    c_locations = list(c_locations)

    u_locations_pt = np.digitize(ujets['pt_uncalib'], pt_bins) - 1
    u_locations_eta = np.digitize(ujets['abs_eta_uncalib'], eta_bins) - 1
    u_locations = zip(u_locations_pt, u_locations_eta)
    u_locations = list(u_locations)

    c_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
    b_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
    u_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
    print('Grouping the bins')
    for i, x in enumerate(c_locations):
        #print ('i: ', i , 'x: ', x )
        c_loc_indices[x].append(i)

    for i, x in enumerate(b_locations):
        b_loc_indices[x].append(i)

    for i, x in enumerate(u_locations):
        u_loc_indices[x].append(i)

    cjet_indices = []
    bjet_indices = []
    ujet_indices = []
    print('Matching the bins for all flavours')
    for pt_bin_i in range(len(pt_bins) - 1):
        for eta_bin_i in range(len(eta_bins) - 1):
            loc = (pt_bin_i, eta_bin_i)

            nbjets = int(histvals_b[eta_bin_i][pt_bin_i])
            ncjets = int(histvals_c[eta_bin_i][pt_bin_i])
            nujets = int(histvals_u[eta_bin_i][pt_bin_i])

            njets = min([nbjets, ncjets, nujets])
            c_indices_for_bin = c_loc_indices[loc][0:njets]
            b_indices_for_bin = b_loc_indices[loc][0:njets]
            u_indices_for_bin = u_loc_indices[loc][0:njets]
            cjet_indices += c_indices_for_bin
            bjet_indices += b_indices_for_bin
            ujet_indices += u_indices_for_bin

    cjet_indices.sort()
    bjet_indices.sort()
    ujet_indices.sort()
    return np.array(bjet_indices), np.array(cjet_indices), np.array(ujet_indices)

""" #file_path_o =  ''
file_path_o =  'events_'+even_or_odd+'/'
#file = file_path_o + even_or_odd + '_' + jz_slice + '.h5'
file = file_path_o + 'outfile_0.h5'
df_Zprime = h5py.File(file, "r")['jets'][:] ### all
#df_file = pd.DataFrame(h5py.File(file, "r")['jets'][:]) ### all 

samp_start = 1 
samp_e = 133
for i in range(samp_start,samp_e):
    #temp_path = file_path_o + even_or_odd + '_' + str(i) + '.h5'
    temp_path = file_path_o + 'outfile' + '_' + str(i) + '.h5'
    temp_data_frame = h5py.File(temp_path, "r")['jets'][:]
    df_Zprime = np.concatenate((df_Zprime, temp_data_frame))
 """

print('Grab 1st file')

file = 'events_data/data18.00365502.physics_HardProbes.h5'
df_Zprime = h5py.File(file, "r")['jets'][:] ### all


""" samp_start = 1 
samp_e = 4
for i in range(samp_start,samp_e):
    temp_path = 'data' + str(i) + '.h5'
    temp_data_frame = h5py.File(temp_path, "r")['jets'][:]
    df_Zprime = np.concatenate((df_Zprime, temp_data_frame)) """
print('Grab 2nd file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00365573.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 3rd file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00365752.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 4th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00365914.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 5th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366268.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 6th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366337.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 7th file')
""" df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366413.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 8th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366627.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 9th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366691.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 10th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366805.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 11th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366860.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 12th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366919.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 13th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00366994.physics_HardProbes.h5', "r")['jets'][:]))
print('Grab 14th file')
df_Zprime = np.concatenate((df_Zprime,h5py.File('events_data/data18.00367165.physics_HardProbes.h5', "r")['jets'][:])) """

print('Done with files.')

print (type(df_Zprime))


binning = {"pt_uncalib": np.linspace(0, 800000, 81),#/1000,
               "abs_eta_uncalib": np.linspace(0, 2.5, 26)}
var = "pt_uncalib"

""" plt.hist(df_Zprime[df_Zprime['HadronConeExclTruthLabelID']==5][var]/1000, binning[var]/1000, histtype='step',
         label=["b-jets"],
         stacked=False,
         fill=False,
         linewidth=2, alpha=0.8)

plt.xlabel('$p_T$ [GeV]',fontsize=12)
plt.ylabel('# jets',fontsize=12)
plt.yscale('log')
plt.legend()
plt.savefig('pdf_files/jet_pT_'+even_or_odd+'_jz'+jz_slice+'.pdf')  """


bjets = df_Zprime
""" bjets = df_Zprime[df_Zprime['HadronConeExclTruthLabelID']==5]
cjets = df_Zprime[df_Zprime['HadronConeExclTruthLabelID']==4]
ujets = df_Zprime[df_Zprime['HadronConeExclTruthLabelID']==0] """

print('b: ', bjets.size)
""" print('c: ', cjets.size)
print('u: ', ujets.size) """

#bjet_indices, cjet_indices, ujet_indices = DownSampling(bjets, cjets, ujets)
#bjet_indices = bjets

#bjets = bjets[bjet_indices]
""" cjets = cjets[cjet_indices]
ujets = ujets[ujet_indices] """

""" plt.hist(bjets[var]/1000, binning[var]/1000, histtype='step',
         label=["b-jets"],
         stacked=False,
         fill=False,
         linewidth=2, alpha=0.8)
plt.hist(cjets[var]/1000, binning[var]/1000, histtype='step',
         label=["c-jets"],
         stacked=False,
         fill=False,
         linewidth=2, alpha=0.8)
plt.hist(ujets[var]/1000, binning[var]/1000, histtype='step',
         label=["light-jets"],
         stacked=False,
         fill=False,
         linewidth=2, alpha=0.8)
plt.xlabel('p$_T$ [GeV]')
plt.ylabel('# jets')
plt.yscale('log')
plt.legend()

plt.savefig('pdf_files/downsampled_jet_pT_'+even_or_odd+'_jz'+jz_slice+'.pdf') """


fig, ax = plt.subplots(10, 5, figsize=(35, 35))
nbins = 50

with open("HI_DL1r_Variables.json") as vardict:
    variablelist = json.load(vardict)[:]
variablelist.remove("HadronConeExclTruthLabelID")

####################### Add the probabilities
""" variablelist.append("pu")
variablelist.append("pc")
variablelist.append("pb") """


def Gen_default_dict(scale_dict):
    ###Generates default value dictionary from scale/shift dictionary.###
    default_dict = {}
    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        default_dict[elem['name']] = elem['default']
    return default_dict

with open("HI_DL1r_Variables.json") as vardict:
    var_names = json.load(vardict)[:]
print(var_names)


#def ScaleVariables(bjets, cjets, ujets):
def ScaleVariables(bjets):
    with open("params_MC16D-ext_2018-PFlow_70-8M_mu.json", 'r') as infile:
        scale_dict = json.load(infile)
    bjets = pd.DataFrame(bjets)
    print(bjets)
    """ cjets = pd.DataFrame(cjets)
    ujets = pd.DataFrame(ujets) """
    bjets.replace([np.inf, -np.inf], np.nan, inplace=True)
    """ cjets.replace([np.inf, -np.inf], np.nan, inplace=True)
    ujets.replace([np.inf, -np.inf], np.nan, inplace=True) """
    # Replace NaN values with default values from default dictionary
    default_dict = Gen_default_dict(scale_dict)
    bjets.fillna(default_dict, inplace=True)
    """ cjets.fillna(default_dict, inplace=True)
    ujets.fillna(default_dict, inplace=True) """
    # scale and shift distribution
    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        if elem['name'] not in var_names:
            continue
        else:

            bjets[elem['name']] = ((bjets[elem['name']] - elem['shift']) /
                                  elem['scale'])
            """ cjets[elem['name']] = ((cjets[elem['name']] - elem['shift']) /
                                  elem['scale'])
            ujets[elem['name']] = ((ujets[elem['name']] - elem['shift']) /
                                  elem['scale']) """
    #bjets = bjets.astype(np.float)
    #cjets = bjets.astype(np.float)
    #ujets = bjets.astype(np.float)
    return bjets.to_records(index=False, column_dtypes='float64')#, cjets.to_records(index=False, column_dtypes='float64'), ujets.to_records(index=False, column_dtypes='float64')


#bjets_scaled, cjets_scaled, ujets_scaled = ScaleVariables(bjets, cjets, ujets)
bjets_scaled = ScaleVariables(bjets)

print('Scaled: ', bjets_scaled)

if 'HadronConeExclTruthLabelID' in var_names:
    var_names.remove('HadronConeExclTruthLabelID')



print('concatenating flavour samples')
X_test = bjets_scaled
X_test = repack_fields(X_test[var_names])
X_test = X_test.view(np.float64).reshape(X_test.shape + (-1,))

""" X_train = np.concatenate((ujets_scaled, cjets_scaled, bjets_scaled))
y_train = np.concatenate((np.zeros(len(ujets_scaled)),
                          np.ones(len(cjets_scaled)),
                          2 * np.ones(len(bjets_scaled))))


Y_train = np_utils.to_categorical(y_train, 3)

X_train = repack_fields(X_train[var_names])
X_train = X_train.view(np.float64).reshape(X_train.shape + (-1,)) """

###########Normalizing the values of X_train
""" X_train = np.array(X_train)

for i in range(X_train.shape[1]):
    min_val = np.min(X_train[:,i])
    X_train[:,i] = np.subtract(X_train[:,i], min_val)
    max_val = np.max(X_train[:,i])
    X_train[:,i] = np.divide(X_train[:,i], max_val) """

""" rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)

print ('X_train',type(X_train), 'len(): ', len(X_train))
print ('Y_train',type(Y_train), 'len(): ', len(Y_train))

outfile_name = ''
if (even_or_odd == 'even'): outfile_name = file_path_o+'training_'+even_or_odd+'_jz'+jz_slice+'.h5'
else: outfile_name = file_path_o+'testing_'+even_or_odd+'_jz'+jz_slice+'.h5'

print('train/valid output: ',outfile_name)
h5f = h5py.File(outfile_name, 'w')
if (even_or_odd == 'even'):
    h5f.create_dataset('X_train', data=X_train, compression='gzip')
    h5f.create_dataset('Y_train', data=Y_train, compression='gzip')
    h5f.close()
else:
    h5f.create_dataset('X_test', data=X_train, compression='gzip')
    h5f.create_dataset('Y_test', data=Y_train, compression='gzip')
    h5f.close()    """
print('Outfile made: ')
outfile_name = 'real_data_testing_wow_jz'+jz_slice+'.h5'
h5f = h5py.File(outfile_name, 'w')
h5f.create_dataset('X_test', data=X_test, compression='gzip')
h5f.close()
print('Outfile made')

varcounter = -1
for i, axobjlist in enumerate(ax):
      for j, axobj in enumerate(axobjlist):
        varcounter+=1
        if varcounter < len(variablelist):
            var = variablelist[varcounter]
            print('var',var,bjets_scaled[var][10])
            b = pd.DataFrame({var: bjets_scaled[var]})
            """ c = pd.DataFrame({var: cjets_scaled[var]})
            u = pd.DataFrame({var: ujets_scaled[var]}) """
            b.replace([np.inf, -np.inf], np.nan, inplace=True)
            """ c.replace([np.inf, -np.inf], np.nan, inplace=True)
            u.replace([np.inf, -np.inf], np.nan, inplace=True)
 """
            b = b.dropna()
            """ c = c.dropna()
            u = u.dropna() """

            minval = np.amin(b[var])
            maxval = np.amax(b[var])*1.4
            """ if 'pt' in var:
                maxval = np.percentile(u[var],99.99)
            else:
                maxval = max([np.amax(u[var]), np.amax(b[var])])*1.4
            binning = np.linspace(minval,maxval,nbins)
            ###### Fixing Limits of the last 2 histograms
            if varcounter == 43 or varcounter == 44:
                minval = 0
                maxval = 1.4 """
            binning = np.linspace(minval,maxval,nbins)

            axobj.hist(b[var],binning,histtype=u'step', color='orange',label='b-jets',density=1)
            """ axobj.hist(c[var],binning,histtype=u'step', color='b',label='c-jets',density=1)
            axobj.hist(u[var],binning,histtype=u'step', color='g',label='u-jets',density=1) """

            axobj.legend()
            axobj.set_yscale('log',nonpositive='clip')
            axobj.set_title(variablelist[varcounter])

        else:
            axobj.axis('off')

plt.tight_layout()
#plt.savefig('pdf_files/DL1-variables_'+even_or_odd+'_jz'+jz_slice+'.pdf', transparent=True)
plt.savefig('pdf_files/RealData_2'+'.pdf', transparent=True)