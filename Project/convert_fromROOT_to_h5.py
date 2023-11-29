import sys
import uproot3 as up
import numpy as np
import h5py
import pandas as pd
from Variable_mapping import mapping, var_conv_oldDl1
from MV2_defaults import default_values2
import argparse


def GetArgs():

    """parse arguments"""

    parser = argparse.ArgumentParser(
        description="ROOT to hdf5 converter",
        usage="python convert_fromROOT.py <options>"
        )

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "--input", action="store", dest="input_files",
        metavar="input files", required=True, nargs='+',
        help="path to input ROOT files to merge using bash synthax"
    )

    required.add_argument(
        "--output", action="store", dest="output",
        metavar="[path/to/filename]", required=True,
        help="path to output hdf5 file"
        )

    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "--events", action="store", dest="events", type=int,
        required=False, default=1e6, help="Amount of events to process"
        )

    optional.add_argument(
        "--single-file", action="store_true", dest="single",
        help="Option to save all events in one file"
        )
    split = parser.add_mutually_exclusive_group()
    split.add_argument('--even', action='store_true')
    split.add_argument('--odd', action='store_true')
    category = parser.add_mutually_exclusive_group()
    category.add_argument('--bjets', action='store_true')
    category.add_argument('--cjets', action='store_true')
    category.add_argument('--ujets', action='store_true')
    return parser.parse_args()


def FindCheck(jetVec):
    default_location = np.argwhere(np.isnan(jetVec))
    jet_feature_check = np.zeros(len(jetVec))
    jet_feature_check[default_location] = 1
    return jet_feature_check


def GetTree(file_name, add_cuts):
    """Retrieves the events in the TTree with uproot and returns them as
    a pandas DataFrame."""
    var_list = list(mapping.keys())
    # var_list += ['jet_pt', 'jet_eta', 'jet_JVT', 'jet_aliveAfterOR']
    #var_list += ['jet_pt', 'jet_eta' ]
    tree = up.open(file_name)['bTag_DFAntiKt4HIJets']
    # tree = up.open(file_name+':bTag_AntiKt4HIJets')
    # print(tree.keys())
    # print(var_list)
    #df = tree.arrays(var_list, library='pd')
    df = tree.pandas.df(var_list)
    # Apply jet quality cuts
    # df.query("jet_pt>20e3 & abs(jet_eta)<2.5 & (abs(jet_eta)>2.4 |\
    #          jet_pt>60e3 | jet_JVT>0.2) & (jet_aliveAfterOR ==True)",
    #          inplace=True)
    print(df)
    print(type(df))
    df.query("jet_pt>50e3 & abs(jet_eta)<2.5 ",
             inplace=True)
    df.rename(index=str, columns=mapping, inplace=True)

    if add_cuts != "":
        df.query(add_cuts, inplace=True)
    # changing eta to absolute eta
    df['abs_eta_uncalib'] = df['abs_eta_uncalib'].abs()
    # Replacing default values with this synthax
    # df.replace({'A': {0: 100, 4: 400}})
    rep_dict = {}
    for key, val in default_values2.items():
        if key in list(var_conv_oldDl1.keys()):
            replacer = {}
            for elem in val:
                replacer[elem] = np.nan
            rep_dict[var_conv_oldDl1[key]] = replacer
    df.replace(rep_dict, inplace=True)

    # Generating default flags
    df['JetFitter_isDefaults'] = FindCheck(df['JetFitter_mass'].values)
    df['SV1_isDefaults'] = FindCheck(df['SV1_masssvx'].values)
    df['IP2D_isDefaults'] = FindCheck(df['IP2D_bu'].values)
    df['IP3D_isDefaults'] = FindCheck(df['IP3D_bu'].values)
    df['secondaryVtx_isDefaults'] = FindCheck(df['secondaryVtx_nTrks'].values)
    
    return df


def __run():
    args = GetArgs()
    events = 0
    df_out = None
    # additional cuts on eventNumber and class label
    add_cuts = ""
    parity_cut = ""
    if args.even:
        add_cuts = "(eventNumber % 2 == 0)"
    elif args.odd:
        add_cuts = "(eventNumber % 2 == 1)"
    if args.bjets:
        parity_cut = "(HadronConeExclTruthLabelID == 5)"
    elif args.cjets:
        parity_cut = "(HadronConeExclTruthLabelID == 4)"
    elif args.ujets:
        parity_cut = "(HadronConeExclTruthLabelID == 0)"
    if parity_cut != "":
        add_cuts += "& %s" % parity_cut

    for i, file in enumerate(args.input_files):
        events_rest = int(args.events - events)
        if events_rest <= 0:
            break
        # print(events_rest, "events more to process")
        sys.stdout.write('\r')
        # the exact output you're looking for:
        j = (events + 1) / args.events
        sys.stdout.write("%i/%i  [%-20s] %d%%" % (events, args.events,
                                                  '='*int(20*j), 100*j))
        sys.stdout.flush()
        df = GetTree(file, add_cuts)
        print ("type of df ", type(df) )
        print ('size of df ', df.size) 
        print ('shape of df', df.shape)
        print ('df before save ', df)
        if args.single is False:
            #outfile_name = "%s/ttbar_PFlow-newIPtag-%i.h5" % (args.output, i)
            outfile_name = "%s" % (args.output)
            h5f = h5py.File(outfile_name, 'w')
            h5f.create_dataset('jets', data=df)
            # h5f.create_dataset('jets', data=df.sample(frac=1).to_records(index=False)[:int(args.events)])
            # h5f.create_dataset('jets', data=df.to_records(index=False)[:])
            h5f.close()
        else:
            if df_out is None:
                df_out = df
            else:
                df_out = pd.concat([df_out, df])
        events += len(df)
    print("")
    # outfile_name = "%s/ttbar_PFlow-newIPtag-merged-even-bjets.h5" % (args.output)
    h5f = h5py.File(args.output, 'w')
    #h5f.create_dataset('jets', data=df_out.sample(frac=1).to_records(index=False)[:int(args.events)])
    #h5f.create_dataset('jets', data=df)
    h5f.create_dataset('jets', data=df.to_records(index=False)[:])
    h5f.close()


__run()
