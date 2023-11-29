import numpy as np
from numpy.lib.recfunctions import repack_fields
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import json

def plotBin(myModel, X_test, Y_test, returnDisc=False, fc=0.08, fig_name='dl1_discriminant.pdf', jz='incl', ver='77',  file = 'something', name = 'model_name'):
    def sigBkgEff1():
        '''
        Given a model, make the histograms of the model outputs to get the ROC curves.

        Input:
            myModel: A keras model
            X_test: Model inputs of the test set
            y_test: Truth labels for the test set
            returnDisc: If True, also return the raw discriminant
            fc: The amount by which to weight the c-jet prob in the disc. The
                default value of 0.07 corresponds to the fraction of c-jet bkg
                in ttbar.

        Output:
            effs: A list with 3 entries for the l, c, and b effs
            disc: b-tagging discriminant (will only be returned if returnDisc is True)
        '''
        #print("sig 1 returnDisc " ,returnDisc)
        #print("sig 1 fc " ,fc)
        #print("sig 1 fig_name " ,fig_name)
        #print("sig 1 jz " ,jz)

        print('Filename :', file)

        # Evaluate the performance with the ROC curves!
        predictions = myModel.predict(X_test,verbose=True)

        ######################################################################## Different Centralities
        with open("HI_DL1r_Variables.json") as vardict:
            variablelist = json.load(vardict)[:]
        variablelist.remove("HadronConeExclTruthLabelID")

        ####################### Add the predictions
        #variablelist.append("pu")
        #variablelist.append("pc")
        #variablelist.append("pb")

        #print(X_test)

        with open("HI_DL1r_Variables.json") as vardict:
            var_names = json.load(vardict)[:]
        #print(var_names)
        if 'HadronConeExclTruthLabelID' in var_names:
            var_names.remove('HadronConeExclTruthLabelID')


        def ScaleVariables(bjets):
            with open("params_MC16D-ext_2018-PFlow_70-8M_mu.json", 'r') as infile:
                scale_dict = json.load(infile)
            bjets = pd.DataFrame(bjets)
            bjets.rename(columns = {41:'FCalEt', 1:'pt_uncalib'}, inplace = True)
            print(bjets)
            # unscale and shift distribution
            for elem in scale_dict:
                if 'isDefaults' in elem['name']:
                    continue
                if elem['name'] not in var_names:
                    continue
                if elem['name'] == 'FCalEt' or elem['name'] == 'pt_uncalib':
                #else:
                    bjets[elem['name']] = (bjets[elem['name']] + elem['shift'])*elem['scale']
            return bjets#bjets.to_records(index=False, column_dtypes='float64')

        if 'HadronConeExclTruthLabelID' in var_names:
            var_names.remove('HadronConeExclTruthLabelID')

        X_test_scaled = ScaleVariables(X_test)
        #print('X_test Scaled:', X_test_scaled)
        
        DF = pd.DataFrame(X_test_scaled)
        DF['pu'] = predictions[:,0]
        DF['pc'] = predictions[:,1]
        DF['pb'] = predictions[:,2]

        """ DF['True pu'] = Y_test[:,0]
        DF['True pc'] = Y_test[:,1]
        DF['True pb'] = Y_test[:,2] """

        DF['Label'] = 0*Y_test[:,0] + Y_test[:,1] + 2*Y_test[:,2]

        #print(DF)


        ########################################### Cuts for each centrality
        #0-10%
        DF_cent_0_10 = DF[(2.98931<DF['FCalEt'])&(DF['FCalEt']<5.5)]
        #10-20%
        DF_cent_10_20 = DF[(2.04651<DF['FCalEt'])&(DF['FCalEt']<2.98931)]
        #20-30%
        DF_cent_20_30 = DF[(1.36875<DF['FCalEt'])&(DF['FCalEt']<2.04651)]
        #30-40%
        DF_cent_30_40 = DF[(0.87541<DF['FCalEt'])&(DF['FCalEt']<1.36875)]
        #40-50%
        DF_cent_40_50 = DF[(0.525092<DF['FCalEt'])&(DF['FCalEt']<0.87541)]
        #50-60%
        DF_cent_50_60 = DF[(0.289595<DF['FCalEt'])&(DF['FCalEt']<0.525092)]
        #60-70%
        DF_cent_60_70 = DF[(0.14414<DF['FCalEt'])&(DF['FCalEt']<0.289595)]
        #70-80%
        DF_cent_70_80 = DF[(0.063719<DF['FCalEt'])&(DF['FCalEt']<0.14414)]
        #50-80%
        DF_perif = DF[(0.063719<DF['FCalEt'])&(DF['FCalEt']<0.289595)]
        #80-100%
        DF_cent_80_100 = DF[(-10.<DF['FCalEt'])&(DF['FCalEt']<0.063719)]

        ########################################## Cuts for pt

        #High-pt
        DF_high_pt = DF[DF['pt_uncalib']>=2000000]
        #Low-pt
        DF_low_pt = DF[DF['pt_uncalib']<2000000]

        #print('High pt', DF_high_pt)

        plt.figure()        
        plt.hist(DF['pu'] , 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_u$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_0_10_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/pu_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close()

        plt.figure()  
        plt.hist(DF['pc'] , 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_c$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_10_20_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/pc_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close()

        plt.figure()  
        plt.hist(DF['pb'] , 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_b$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_70_80_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/pb_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close()

        # To make sure you're not discarding the b-values with high
        # discriminant values that you're good at classifying, use the
        # max from the distribution
 
        disc = np.log(np.divide(predictions[:,2], fc*predictions[:,1] + (1 - fc) * predictions[:,0]))
        #print('predictions[:,2]: ',predictions[:,2])

        #disc_0_10 = np.log(np.divide(DF_cent_0_10['pb'].to_numpy(), fc*DF_cent_0_10['pc'].to_numpy() + (1 - fc) * DF_cent_0_10['pu'].to_numpy()))
        disc_0_10 = np.log(np.divide(DF_perif['pb'].to_numpy(), fc*DF_perif['pc'].to_numpy() + (1 - fc) * DF_perif['pu'].to_numpy()))
        #disc_0_10 = np.log(np.divide(DF_high_pt['pb'].to_numpy(), fc*DF_high_pt['pc'].to_numpy() + (1 - fc) * DF_high_pt['pu'].to_numpy()))
        #disc_0_10 = np.log(np.divide(DF_low_pt['pb'].to_numpy(), fc*DF_low_pt['pc'].to_numpy() + (1 - fc) * DF_low_pt['pu'].to_numpy()))

        '''
        Note: For jets w/o any tracks
        '''

        discMax = np.max(disc)
        discMin = np.min(disc)

        #if np.isfinite(discMax) == 0:
        discMax = 25.
        #if np.isfinite(discMin) == 0:
        discMin = -10.

        myRange=(discMin,discMax)
        nBins = 1000 #350

        plt.figure()
        effs, effs_0_10 = [], []
        yerr = []

        entrs = []
        bws = []
        bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            ix = (np.argmax(Y_test,axis=-1) == output)
            print('np.argmax(Y_test,axis=-1): ', np.argmax(Y_test,axis=-1)[:100]) 
            #0 0 1 1 0 2
            print('ix: ', ix[:100]) 
            #True  True False False  True False  for l
            #False False  True  True False False  for c
            #'''
            # Plot the discriminant output
            nEntries, edges ,_ = plt.hist(disc[ix],alpha=0.5,label='{}-jets'.format(flavor),
                                          bins=nBins, range=myRange, density=1)
            bin_width     = (edges[1]-edges[0])
            bin_widths    = len(nEntries)*[bin_width]
            bin_positions = edges[0:-1]+bin_width/2.0

            entrs.append(nEntries)
            bws.append(bin_widths)
            bps.append(bin_positions)

            wsigma = nEntries ** 0.5

            cx = 0.5 * (edges[1:] + edges[1:])
            #plt.errorbar(cx, nEntries, wsigma, fmt="o")
            yerr.append(wsigma)
            print(' Entries: ',np.sum(nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
            eff = np.add.accumulate(nEntries[::-1]) / np.sum(nEntries)
            effs.append(eff)

        

        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$D = \ln [ p_b / (f_c p_c + (1- f_c)p_l ) ]$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn='pdf_files_uns/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn='pdf_files/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn)

        if (file == '_uns'):
            disc_b_txt_incl = 'pdf_files_uns/disc_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_c_txt_incl = 'pdf_files_uns/disc_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_l_txt_incl = 'pdf_files_uns/disc_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            disc_b_txt_incl = 'pdf_files/disc_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_c_txt_incl = 'pdf_files/disc_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_l_txt_incl = 'pdf_files/disc_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(disc_l_txt_incl, list(zip(bps[0], bws[0], entrs[0],yerr[0])))
        np.savetxt(disc_c_txt_incl, list(zip(bps[1], bws[1], entrs[1],yerr[1])))
        np.savetxt(disc_b_txt_incl, list(zip(bps[2], bws[2], entrs[2],yerr[2])))

        #plt.show()
        plt.close()

        #Label_array = DF_cent_0_10['Label'].to_numpy()
        Label_array = DF_perif['Label'].to_numpy()
        #Label_array = DF_high_pt['Label'].to_numpy()
        #Label_array = DF_low_pt['Label'].to_numpy()

        for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix_0_10 = (pd.Series(T_0_10).argmax(axis=-1).item() == output)
            """ ix_0_10 = (np.argmax(Label_array, axis=-1) == output)
            print('np.argmax(Y_test,axis=-1): ', np.argmax(Label_array, axis=-1)[:100])  """
            ix_0_10 = (Label_array == output)
            #print('np.argmax(Y_test,axis=-1): ', Label_array[:100]) 
            #0 0 1 1 0 2
            #print('ix_0_10: ', ix_0_10[:100])  
            #True  True False False  True False  for l
            #False False  True  True False False  for c
            #'''
            # Plot the discriminant output
            nEntries_0_10, edges_0_10 ,_0_10 = plt.hist(disc_0_10[ix_0_10],alpha=0.5,label='{}-jets'.format(flavor),
                                          bins=nBins, range=myRange, density=1)
          
            print(' Entries: ',np.sum(nEntries_0_10), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
            eff = np.add.accumulate(nEntries_0_10[::-1]) / np.sum(nEntries_0_10)
            effs_0_10.append(eff)

        # probabilities
        pb = predictions[:,2]
        pb_Max = np.max(pb)
        pb_Min = np.min(pb)
        pb_myRange=(pb_Min,pb_Max)
        plt.figure()
        pb_yerr = []

        pb_entrs = []
        pb_bws = []
        pb_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            ix = (np.argmax(Y_test,axis=-1) == output)

            # Plot the discriminant output
            pb_nEntries, pb_edges ,_ = plt.hist(pb[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pb_myRange, density=1)
            pb_bin_width     = (pb_edges[1]-pb_edges[0])
            pb_bin_widths    = len(pb_nEntries)*[pb_bin_width]
            pb_bin_positions = pb_edges[0:-1]+pb_bin_width/2.0

 
            pb_entrs.append(pb_nEntries)
            pb_bws.append(pb_bin_widths)
            pb_bps.append(pb_bin_positions)

            pb_wsigma = pb_nEntries ** 0.5
            # draw data
            pb_cx = 0.5 * (pb_edges[1:] + pb_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pb_yerr.append(pb_wsigma)
            print(' prob Entries:  ',np.sum(pb_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_b$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files/pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)

        # probabilities
        pc = predictions[:,1]
        pc_Max = np.max(pc)
        pc_Min = np.min(pc)
        pc_myRange=(pc_Min,pc_Max)
        plt.figure()
        pc_yerr = []

        pc_entrs = []
        pc_bws = []
        pc_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            ix = (np.argmax(Y_test,axis=-1) == output)

            # Plot the discriminant output
            pc_nEntries, pc_edges ,_ = plt.hist(pc[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pc_myRange, density=1)
            pc_bin_width     = (pc_edges[1]-pc_edges[0])
            pc_bin_widths    = len(pc_nEntries)*[pc_bin_width]
            pc_bin_positions = pc_edges[0:-1]+pc_bin_width/2.0

 
            pc_entrs.append(pc_nEntries)
            pc_bws.append(pc_bin_widths)
            pc_bps.append(pc_bin_positions)

            pc_wsigma = pc_nEntries ** 0.5
            # draw data
            pc_cx = 0.5 * (pc_edges[1:] + pc_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pc_yerr.append(pc_wsigma)
            print(' prob Entries:  ',np.sum(pc_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_c$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pc_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files/pc_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)
        
        if (file == '_uns'):
            pb_b = 'pdf_files_uns/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_uns/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_uns/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            pb_b = 'pdf_files/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(pb_l, list(zip(pb_bps[0], pb_bws[0], pb_entrs[0],pb_yerr[0])))
        np.savetxt(pb_c, list(zip(pb_bps[1], pb_bws[1], pb_entrs[1],pb_yerr[1])))
        np.savetxt(pb_b, list(zip(pb_bps[2], pb_bws[2], pb_entrs[2],pb_yerr[2])))

        #plt.show()
        plt.close()

        # probabilities
        pu = predictions[:,0]
        pu_Max = np.max(pu)
        pu_Min = np.min(pu)
        pu_myRange=(pu_Min,pu_Max)
        plt.figure()
        pu_yerr = []

        pu_entrs = []
        pu_bws = []
        pu_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            ix = (np.argmax(Y_test,axis=-1) == output)

            # Plot the discriminant output
            pu_nEntries, pu_edges ,_ = plt.hist(pu[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pu_myRange, density=1)
            pu_bin_width     = (pu_edges[1]-pu_edges[0])
            pu_bin_widths    = len(pu_nEntries)*[pu_bin_width]
            pu_bin_positions = pu_edges[0:-1]+pu_bin_width/2.0

 
            pu_entrs.append(pu_nEntries)
            pu_bws.append(pu_bin_widths)
            pu_bps.append(pu_bin_positions)

            pu_wsigma = pu_nEntries ** 0.5
            # draw data
            pu_cx = 0.5 * (pu_edges[1:] + pu_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pu_yerr.append(pu_wsigma)
            print(' prob Entries:  ',np.sum(pu_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_u$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pu_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files/pu_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)
        
        if (file == '_uns'):
            pb_b = 'pdf_files_uns/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_uns/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_uns/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            pb_b = 'pdf_files/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(pb_l, list(zip(pb_bps[0], pb_bws[0], pb_entrs[0],pb_yerr[0])))
        np.savetxt(pb_c, list(zip(pb_bps[1], pb_bws[1], pb_entrs[1],pb_yerr[1])))
        np.savetxt(pb_b, list(zip(pb_bps[2], pb_bws[2], pb_entrs[2],pb_yerr[2])))

        #plt.show()
        plt.close()

        if returnDisc:
            return effs, disc, pb, effs_0_10
        else:
            return effs
    
        

#    print('plot 1 returnDisc ' ,returnDisc)
#    print('plot 1 fcc ' ,fc)
#    print('plot 1 fig_name ' ,fig_name)
    print("plot 1 jz " ,jz)

    (leff, ceff, beff), d, pb, (leff_0_10, ceff_0_10, beff_0_10) = sigBkgEff1()
    #leff, ceff, beff= sigBkgEff1()
    print('probability: ' , pb)
    dl1_leffs, dl1_ceffs, dl1_beffs, dl1_discs = [], [], [], []
    dl1_leffs.append(leff)
    dl1_ceffs.append(ceff)
    dl1_beffs.append(beff)
    dl1_discs.append(d)

    """ dl1_leffs[0] = np.trim_zeros(np.array(dl1_leffs))
    dl1_ceffs[0] = np.trim_zeros(np.array(dl1_ceffs)) """

    dl1_beffs_m = np.ma.masked_equal(np.array(dl1_beffs),0)
    dl1_beffs_m.compressed()
    dl1_leffs_m = np.ma.masked_equal(np.array(dl1_leffs),0)
    dl1_leffs_m.compressed()
    dl1_ceffs_m = np.ma.masked_equal(np.array(dl1_ceffs),0)
    dl1_ceffs_m.compressed()



    print('leffs: ', dl1_leffs_m)

    plt.figure()
    plt.plot(dl1_beffs_m[0], 1. / dl1_leffs_m[0], color='C4', label='l-rej')
    #plt.plot(b_effs, 1./l_rej, color='C2', label='l-rej - pp recommendations')
    # plt.figure()
    plt.plot(dl1_beffs[0], 1. / dl1_ceffs_m[0],"--", color='C4', label='c-rej')
    #plt.plot(b_effs, 1./c_rej, "--", color='C2', label='c-rej - pp recommendations')
    
    plt.ylabel('background-rej')
    
    plt.legend()
    plt.title('Default_0 ')
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.ylim(0,3000)

    if (file == '_uns'):
        ROC_b='pdf_files_uns/ROC'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    else:
        ROC_b='pdf_files/ROC'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    plt.savefig(ROC_b)

    plt.show()
 
    if (file == '_uns'):
        roc_b_vs_l_txt_cent = 'pdf_files_uns/ROC_b_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent = 'pdf_files_uns/ROC_c_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    else:
        roc_b_vs_l_txt_cent = 'pdf_files/ROC_b_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent = 'pdf_files/ROC_c_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    np.savetxt(roc_b_vs_l_txt_cent , np.column_stack((dl1_beffs[0],1./dl1_leffs[0])), delimiter=' ')
    np.savetxt(roc_c_vs_l_txt_cent , np.column_stack((dl1_beffs[0],1./dl1_ceffs[0])), delimiter=' ')

    #############################0-10%
    dl1_leffs_0_10, dl1_ceffs_0_10, dl1_beffs_0_10 = [], [], []
    dl1_leffs_0_10.append(leff_0_10)
    dl1_ceffs_0_10.append(ceff_0_10)
    dl1_beffs_0_10.append(beff_0_10)

    dl1_beffs_m_0_10 = np.ma.masked_equal(np.array(dl1_beffs_0_10),0)
    dl1_beffs_m_0_10.compressed()
    dl1_leffs_m_0_10 = np.ma.masked_equal(np.array(dl1_leffs_0_10),0)
    dl1_leffs_m_0_10.compressed()
    dl1_ceffs_m_0_10 = np.ma.masked_equal(np.array(dl1_ceffs_0_10),0)
    dl1_ceffs_m_0_10.compressed()

    print('leffs_0_10: ', dl1_leffs_m_0_10)    

    ######################0-10%
    plt.figure()
    plt.plot(dl1_beffs_m_0_10[0], 1. / dl1_leffs_m_0_10[0], color='C4', label='l-rej')
    #plt.plot(b_effs, 1./l_rej, color='C2', label='l-rej - pp recommendations')
    # plt.figure()
    plt.plot(dl1_beffs_0_10[0], 1. / dl1_ceffs_m_0_10[0],"--", color='C4', label='c-rej')
    #plt.plot(b_effs, 1./c_rej, "--", color='C2', label='c-rej - pp recommendations')
    
    plt.ylabel('background-rej')
    
    plt.legend()
    #plt.title('Default_0 0-10%')
    #plt.title('Default_0 50-80%')
    plt.title('Default_0 High $p_{t}$')
    #plt.title('Default_0 Low $p_{t}$')
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.ylim(0,3000)

    if (file == '_uns'):
        #ROC_b_0_10='pdf_files_uns/ROC_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        ROC_b_0_10='pdf_files_uns/ROC_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_uns/ROC_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_uns/ROC_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    else:
        #ROC_b_0_10='pdf_files/ROC_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        ROC_b_0_10='pdf_files/ROC_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files/ROC_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files/ROC_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    plt.savefig(ROC_b_0_10)

    plt.show()
 
    if (file == '_uns'):
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    else:
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files/ROC_b_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files/ROC_c_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_b_vs_l_txt_cent_0_10 = 'pdf_files/ROC_b_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent_0_10 = 'pdf_files/ROC_c_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files/ROC_b_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files/ROC_c_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files/ROC_b_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files/ROC_c_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    np.savetxt(roc_b_vs_l_txt_cent_0_10 , np.column_stack((dl1_beffs_0_10[0],1./np.array(dl1_leffs_0_10)[0])), delimiter=' ')
    np.savetxt(roc_c_vs_l_txt_cent_0_10 , np.column_stack((dl1_beffs_0_10[0],1./np.array(dl1_ceffs_0_10)[0])), delimiter=' ')

    return

def plotBinReal(myModel, X_test, returnDisc=False, fc=0.08, fig_name='dl1_discriminant.pdf', jz='incl', ver='77',  file = 'something', name = 'model_name'):
    def sigBkgEff2():
        '''
        Given a model, make the histograms of the model outputs to get the ROC curves.

        Input:
            myModel: A keras model
            X_test: Model inputs of the test set
            y_test: Truth labels for the test set
            returnDisc: If True, also return the raw discriminant
            fc: The amount by which to weight the c-jet prob in the disc. The
                default value of 0.07 corresponds to the fraction of c-jet bkg
                in ttbar.

        Output:
            effs: A list with 3 entries for the l, c, and b effs
            disc: b-tagging discriminant (will only be returned if returnDisc is True)
        '''
        #print("sig 1 returnDisc " ,returnDisc)
        #print("sig 1 fc " ,fc)
        #print("sig 1 fig_name " ,fig_name)
        #print("sig 1 jz " ,jz)

        print('Filename :', file)

        # Evaluate the performance with the ROC curves!
        predictions = myModel.predict(X_test,verbose=True)

        ######################################################################## Different Centralities
        with open("HI_DL1r_Variables.json") as vardict:
            variablelist = json.load(vardict)[:]
        variablelist.remove("HadronConeExclTruthLabelID")

        ####################### Add the predictions
        #variablelist.append("pu")
        #variablelist.append("pc")
        #variablelist.append("pb")

        #print(X_test)

        with open("HI_DL1r_Variables.json") as vardict:
            var_names = json.load(vardict)[:]
        #print(var_names)
        if 'HadronConeExclTruthLabelID' in var_names:
            var_names.remove('HadronConeExclTruthLabelID')


        def ScaleVariables(bjets):
            with open("params_MC16D-ext_2018-PFlow_70-8M_mu.json", 'r') as infile:
                scale_dict = json.load(infile)
            bjets = pd.DataFrame(bjets)
            bjets.rename(columns = {41:'FCalEt', 1:'pt_uncalib'}, inplace = True)
            print(bjets)
            # unscale and shift distribution
            for elem in scale_dict:
                if 'isDefaults' in elem['name']:
                    continue
                if elem['name'] not in var_names:
                    continue
                if elem['name'] == 'FCalEt' or elem['name'] == 'pt_uncalib':
                #else:
                    bjets[elem['name']] = (bjets[elem['name']] + elem['shift'])*elem['scale']
            return bjets#bjets.to_records(index=False, column_dtypes='float64')

        if 'HadronConeExclTruthLabelID' in var_names:
            var_names.remove('HadronConeExclTruthLabelID')

        X_test_scaled = ScaleVariables(X_test)
        #print('X_test Scaled:', X_test_scaled)
        
        DF = pd.DataFrame(X_test_scaled)
        DF['pu'] = predictions[:,0]
        DF['pc'] = predictions[:,1]
        DF['pb'] = predictions[:,2]


        #print(DF)


        ########################################### Cuts for each centrality
        #0-10%
        DF_cent_0_10 = DF[(2.98931<DF['FCalEt'])&(DF['FCalEt']<5.5)]
        #10-20%
        DF_cent_10_20 = DF[(2.04651<DF['FCalEt'])&(DF['FCalEt']<2.98931)]
        #20-30%
        DF_cent_20_30 = DF[(1.36875<DF['FCalEt'])&(DF['FCalEt']<2.04651)]
        #30-40%
        DF_cent_30_40 = DF[(0.87541<DF['FCalEt'])&(DF['FCalEt']<1.36875)]
        #40-50%
        DF_cent_40_50 = DF[(0.525092<DF['FCalEt'])&(DF['FCalEt']<0.87541)]
        #50-60%
        DF_cent_50_60 = DF[(0.289595<DF['FCalEt'])&(DF['FCalEt']<0.525092)]
        #60-70%
        DF_cent_60_70 = DF[(0.14414<DF['FCalEt'])&(DF['FCalEt']<0.289595)]
        #70-80%
        DF_cent_70_80 = DF[(0.063719<DF['FCalEt'])&(DF['FCalEt']<0.14414)]
        #50-80%
        DF_perif = DF[(0.063719<DF['FCalEt'])&(DF['FCalEt']<0.289595)]
        #80-100%
        DF_cent_80_100 = DF[(-10.<DF['FCalEt'])&(DF['FCalEt']<0.063719)]

        ########################################## Cuts for pt

        #High-pt
        DF_high_pt = DF[DF['pt_uncalib']>=2000000]
        #Low-pt
        DF_low_pt = DF[DF['pt_uncalib']<2000000]

        #print('High pt', DF_high_pt)

                
        """ plt.hist(DF_cent_0_10['FCalEt'], 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('0-10%',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_0_10_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/cent_0_10_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close()

        plt.hist(DF_cent_10_20['FCalEt'], 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('10-20%',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_10_20_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/cent_10_20_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close()

        plt.hist(DF_cent_70_80['FCalEt'], 50, density=1)
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('70-80%',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            plt.savefig('pdf_files_uns/cent_70_80_'+str(ver)+'_jz'+str(jz)+'.pdf')
        else:
            plt.savefig('pdf_files/cent_70_80_'+str(ver)+'_jz'+str(jz)+'.pdf')
        plt.close() """

        # To make sure you're not discarding the b-values with high
        # discriminant values that you're good at classifying, use the
        # max from the distribution
 
        disc = np.log(np.divide(predictions[:,2], fc*predictions[:,1] + (1 - fc) * predictions[:,0]))
        DF['disc'] = np.log(np.divide(DF['pb'], fc*DF['pc'] + (1 - fc) * DF['pu']))
        print(DF['disc'])
        #print('predictions[:,2]: ',predictions[:,2])

        #disc_0_10 = np.log(np.divide(DF_cent_0_10['pb'].to_numpy(), fc*DF_cent_0_10['pc'].to_numpy() + (1 - fc) * DF_cent_0_10['pu'].to_numpy()))
        disc_0_10 = np.log(np.divide(DF_perif['pb'].to_numpy(), fc*DF_perif['pc'].to_numpy() + (1 - fc) * DF_perif['pu'].to_numpy()))
        #disc_0_10 = np.log(np.divide(DF_high_pt['pb'].to_numpy(), fc*DF_high_pt['pc'].to_numpy() + (1 - fc) * DF_high_pt['pu'].to_numpy()))
        #disc_0_10 = np.log(np.divide(DF_low_pt['pb'].to_numpy(), fc*DF_low_pt['pc'].to_numpy() + (1 - fc) * DF_low_pt['pu'].to_numpy()))

        '''
        Note: For jets w/o any tracks
        '''

        discMax = np.max(disc)
        discMin = np.min(disc)

        #if np.isfinite(discMax) == 0:
        discMax = 25.
        #if np.isfinite(discMin) == 0:
        discMin = -10.

        myRange=(discMin,discMax)
        nBins = 1000 #350

        plt.figure()
        effs, effs_0_10 = [], []
        yerr = []

        entrs = []
        bws = []
        bps = []
        discr = DF['disc']
        discr.replace([np.inf, -np.inf], np.nan, inplace=True)
        #discr.fillna(discr, inplace=True)
        discr = DF['disc'].to_numpy()

        plt.figure()
        plt.hist(discr, 50, density=1)
        #DF['disc'].plot.hist()
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('Discriminant',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn='pdf_files_uns/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn='pdf_files_data/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn)
        plt.close()


        pb = DF['pb'].replace([np.inf, -np.inf], np.nan, inplace=True)
        pb = DF['pb'].to_numpy()

        plt.figure()
        plt.hist(pb, 50, density=1)
        #DF['disc'].plot.hist()
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_b$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn='pdf_files_uns/dl1_pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn='pdf_files_data/dl1_pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn)
        plt.close()

        """ for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix = (np.argmax(Y_test,axis=-1) == output)
            ix = []
            for i in range(len(DF['pb'])):
                if DF['pb']>DF['pu'] and DF['pb']>DF['pc']: ix[i] = (2 == output) 
                if DF['pc']>DF['pu'] and DF['pc']>DF['pb']: ix[i] = (1 == output) 
                else: ix[i] = (0 == output)
            #print('np.argmax(Y_test,axis=-1): ', np.argmax(Y_test,axis=-1)[:100]) 
            #0 0 1 1 0 2
            print('ix: ', ix[:100]) 
            #True  True False False  True False  for l
            #False False  True  True False False  for c
            #'''
            # Plot the discriminant output
            nEntries, edges ,_ = plt.hist(disc[ix],alpha=0.5,label='{}-jets'.format(flavor),
                                          bins=nBins, range=myRange, density=1)
            bin_width     = (edges[1]-edges[0])
            bin_widths    = len(nEntries)*[bin_width]
            bin_positions = edges[0:-1]+bin_width/2.0

            entrs.append(nEntries)
            bws.append(bin_widths)
            bps.append(bin_positions)

            wsigma = nEntries ** 0.5

            cx = 0.5 * (edges[1:] + edges[1:])
            #plt.errorbar(cx, nEntries, wsigma, fmt="o")
            yerr.append(wsigma)
            print(' Entries: ',np.sum(nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
            eff = np.add.accumulate(nEntries[::-1]) / np.sum(nEntries)
            effs.append(eff)


        for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix_0_10 = (pd.Series(T_0_10).argmax(axis=-1).item() == output)
            #ix_0_10 = (np.argmax(Label_array, axis=-1) == output)
            #print('np.argmax(Y_test,axis=-1): ', np.argmax(Label_array, axis=-1)[:100])
            #ix_0_10 = (Label_array == output)
            ix_0_10 = []
            for i in range(len(DF_perif['pb'])):
                if DF_perif['pb']>DF_perif['pu'] and DF_perif['pb']>DF_perif['pc']: ix_0_10[i] = (2 == output) 
                if DF_perif['pc']>DF_perif['pu'] and DF_perif['pc']>DF_perif['pb']: ix_0_10[i] = (1 == output) 
                else: ix_0_10[i] = (0 == output)
            #print('np.argmax(Y_test,axis=-1): ', Label_array[:100]) 
            #0 0 1 1 0 2
            #print('ix_0_10: ', ix_0_10[:100])  
            #True  True False False  True False  for l
            #False False  True  True False False  for c
            #'''
            # Plot the discriminant output
            nEntries_0_10, edges_0_10 ,_0_10 = plt.hist(disc_0_10[ix_0_10],alpha=0.5,label='{}-jets'.format(flavor),
                                          bins=nBins, range=myRange, density=1)
          
            print(' Entries: ',np.sum(nEntries_0_10), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
            eff = np.add.accumulate(nEntries_0_10[::-1]) / np.sum(nEntries_0_10)
            effs_0_10.append(eff)

        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$D = \ln [ p_b / (f_c p_c + (1- f_c)p_l ) ]$',fontsize=14)
        plt.ylabel('Normalized counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn='pdf_files_uns/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn='pdf_files_data/dl1_discriminant_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn)

        if (file == '_uns'):
            disc_b_txt_incl = 'pdf_files_uns/disc_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_c_txt_incl = 'pdf_files_uns/disc_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_l_txt_incl = 'pdf_files_uns/disc_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            disc_b_txt_incl = 'pdf_files_data/disc_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_c_txt_incl = 'pdf_files_data/disc_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            disc_l_txt_incl = 'pdf_files_data/disc_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(disc_l_txt_incl, list(zip(bps[0], bws[0], entrs[0],yerr[0])))
        np.savetxt(disc_c_txt_incl, list(zip(bps[1], bws[1], entrs[1],yerr[1])))
        np.savetxt(disc_b_txt_incl, list(zip(bps[2], bws[2], entrs[2],yerr[2])))

        #plt.show()
        plt.close()

        # probabilities
        pb = predictions[:,2]
        pb_Max = np.max(pb)
        pb_Min = np.min(pb)
        pb_myRange=(pb_Min,pb_Max)
        plt.figure()
        pb_yerr = []

        pb_entrs = []
        pb_bws = []
        pb_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix = (np.argmax(Y_test,axis=-1) == output)
            ix = []
            for i in range(len(DF['pb'])):
                if DF['pb']>DF['pu'] and DF['pb']>DF['pc']: ix[i] = (2 == output) 
                if DF['pc']>DF['pu'] and DF['pc']>DF['pb']: ix[i] = (1 == output) 
                else: ix[i] = (0 == output)

            # Plot the discriminant output
            pb_nEntries, pb_edges ,_ = plt.hist(pb[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pb_myRange, density=1)
            pb_bin_width     = (pb_edges[1]-pb_edges[0])
            pb_bin_widths    = len(pb_nEntries)*[pb_bin_width]
            pb_bin_positions = pb_edges[0:-1]+pb_bin_width/2.0

 
            pb_entrs.append(pb_nEntries)
            pb_bws.append(pb_bin_widths)
            pb_bps.append(pb_bin_positions)

            pb_wsigma = pb_nEntries ** 0.5
            # draw data
            pb_cx = 0.5 * (pb_edges[1:] + pb_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pb_yerr.append(pb_wsigma)
            print(' prob Entries:  ',np.sum(pb_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_b$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files_data/pb_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)

        # probabilities
        pc = predictions[:,1]
        pc_Max = np.max(pc)
        pc_Min = np.min(pc)
        pc_myRange=(pc_Min,pc_Max)
        plt.figure()
        pc_yerr = []

        pc_entrs = []
        pc_bws = []
        pc_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix = (np.argmax(Y_test,axis=-1) == output)
            ix = []
            for i in range(len(DF['pb'])):
                if DF['pb']>DF['pu'] and DF['pb']>DF['pc']: ix[i] = (2 == output) 
                if DF['pc']>DF['pu'] and DF['pc']>DF['pb']: ix[i] = (1 == output) 
                else: ix[i] = (0 == output)

            # Plot the discriminant output
            pc_nEntries, pc_edges ,_ = plt.hist(pc[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pc_myRange, density=1)
            pc_bin_width     = (pc_edges[1]-pc_edges[0])
            pc_bin_widths    = len(pc_nEntries)*[pc_bin_width]
            pc_bin_positions = pc_edges[0:-1]+pc_bin_width/2.0

 
            pc_entrs.append(pc_nEntries)
            pc_bws.append(pc_bin_widths)
            pc_bps.append(pc_bin_positions)

            pc_wsigma = pc_nEntries ** 0.5
            # draw data
            pc_cx = 0.5 * (pc_edges[1:] + pc_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pc_yerr.append(pc_wsigma)
            print(' prob Entries:  ',np.sum(pc_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_c$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pc_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files_data/pc_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)
        
        if (file == '_uns'):
            pb_b = 'pdf_files_uns/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_uns/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_uns/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            pb_b = 'pdf_files_data/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_data/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_data/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(pb_l, list(zip(pb_bps[0], pb_bws[0], pb_entrs[0],pb_yerr[0])))
        np.savetxt(pb_c, list(zip(pb_bps[1], pb_bws[1], pb_entrs[1],pb_yerr[1])))
        np.savetxt(pb_b, list(zip(pb_bps[2], pb_bws[2], pb_entrs[2],pb_yerr[2])))

        #plt.show()
        plt.close()

        # probabilities
        pu = predictions[:,0]
        pu_Max = np.max(pu)
        pu_Min = np.min(pu)
        pu_myRange=(pu_Min,pu_Max)
        plt.figure()
        pu_yerr = []

        pu_entrs = []
        pu_bws = []
        pu_bps = []

        for output, flavor in zip([0,1,2], ['l','c','b']):

            #ix = (np.argmax(Y_test,axis=-1) == output)
            ix = []
            for i in range(len(DF['pb'])):
                if DF['pb'][i]>DF['pu'] and DF['pb']>DF['pc']: ix[i] = (2 == output) 
                if DF['pc']>DF['pu'] and DF['pc']>DF['pb']: ix[i] = (1 == output) 
                else: ix[i] = (0 == output)

            # Plot the discriminant output
            pu_nEntries, pu_edges ,_ = plt.hist(pu[ix],alpha=0.5,label='prob. {}-jets'.format(flavor),
                                          bins=100, range=pu_myRange, density=1)
            pu_bin_width     = (pu_edges[1]-pu_edges[0])
            pu_bin_widths    = len(pu_nEntries)*[pu_bin_width]
            pu_bin_positions = pu_edges[0:-1]+pu_bin_width/2.0

 
            pu_entrs.append(pu_nEntries)
            pu_bws.append(pu_bin_widths)
            pu_bps.append(pu_bin_positions)

            pu_wsigma = pu_nEntries ** 0.5
            # draw data
            pu_cx = 0.5 * (pu_edges[1:] + pu_edges[1:])
            #plt.errorbar(pb_cx, pb_nEntries, pb_wsigma, fmt="o")
            pu_yerr.append(pu_wsigma)
            print(' prob Entries:  ',np.sum(pu_nEntries), 'flavor: ',flavor)
            '''
            nEntries is just a sum of the weight of each bin in the histogram.

            Since high Db scores correspond to more b-like jets, compute the cummulative density function
            from summing from high to low values, this is why we reverse the order of the bins in nEntries
            using the "::-1" numpy indexing.
            '''
        plt.legend()
        plt.title('Default_0 ')
        plt.xlabel('$p_u$',fontsize=14)
        plt.ylabel('"Normalized" counts')
        plt.yscale("log")
        if (file == '_uns'):
            fn_pb='pdf_files_uns/pu_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        else:
            fn_pb='pdf_files_data/pu_v'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        plt.savefig(fn_pb)
        
        if (file == '_uns'):
            pb_b = 'pdf_files_uns/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_uns/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_uns/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        else:
            pb_b = 'pdf_files_data/pb_b_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_c = 'pdf_files_data/pb_c_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
            pb_l = 'pdf_files_data/pb_l_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'

        np.savetxt(pb_l, list(zip(pb_bps[0], pb_bws[0], pb_entrs[0],pb_yerr[0])))
        np.savetxt(pb_c, list(zip(pb_bps[1], pb_bws[1], pb_entrs[1],pb_yerr[1])))
        np.savetxt(pb_b, list(zip(pb_bps[2], pb_bws[2], pb_entrs[2],pb_yerr[2])))

        #plt.show()
        plt.close() """

        if returnDisc:
            return disc, pb#, effs_0_10, effs
        else:
            return effs
    
        

#    print('plot 1 returnDisc ' ,returnDisc)
#    print('plot 1 fcc ' ,fc)
#    print('plot 1 fig_name ' ,fig_name)
    print("plot 1 jz " ,jz)

    d, pb = sigBkgEff2()
    #(leff, ceff, beff), d, pb, (leff_0_10, ceff_0_10, beff_0_10) = sigBkgEff2()
    #leff, ceff, beff= sigBkgEff1()
    print('probability: ' , pb)
    dl1_leffs, dl1_discs = [], []
    #dl1_leffs.append(eff)
    dl1_discs.append(d)

    #dl1_leffs[0] = np.trim_zeros(np.array(dl1_leffs))
    #dl1_ceffs[0] = np.trim_zeros(np.array(dl1_ceffs)) 

    """ dl1_beffs_m = np.ma.masked_equal(np.array(dl1_beffs),0)
    dl1_beffs_m.compressed()
    #dl1_leffs_m = np.ma.masked_equal(np.array(dl1_leffs),0)
    #dl1_leffs_m.compressed()
    dl1_ceffs_m = np.ma.masked_equal(np.array(dl1_ceffs),0)
    dl1_ceffs_m.compressed()



    print('leffs: ', dl1_leffs_m)

    plt.figure()
    plt.plot(dl1_beffs_m[0], 1. / dl1_leffs_m[0], color='C4', label='l-rej')
    #plt.plot(b_effs, 1./l_rej, color='C2', label='l-rej - pp recommendations')
    # plt.figure()
    plt.plot(dl1_beffs[0], 1. / dl1_ceffs_m[0],"--", color='C4', label='c-rej')
    #plt.plot(b_effs, 1./c_rej, "--", color='C2', label='c-rej - pp recommendations')
    
    plt.ylabel('background-rej')
    
    plt.legend()
    plt.title('Default_0 ')
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.ylim(0,3000)

    if (file == '_uns'):
        ROC_b='pdf_files_uns/ROC'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    else:
        ROC_b='pdf_files_data/ROC'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    plt.savefig(ROC_b)

    plt.show()
 
    if (file == '_uns'):
        roc_b_vs_l_txt_cent = 'pdf_files_uns/ROC_b_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent = 'pdf_files_uns/ROC_c_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    else:
        roc_b_vs_l_txt_cent = 'pdf_files_data/ROC_b_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent = 'pdf_files_data/ROC_c_vs_lrej_v'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    np.savetxt(roc_b_vs_l_txt_cent , np.column_stack((dl1_beffs[0],1./dl1_leffs[0])), delimiter=' ')
    np.savetxt(roc_c_vs_l_txt_cent , np.column_stack((dl1_beffs[0],1./dl1_ceffs[0])), delimiter=' ')

    #############################0-10%
    dl1_leffs_0_10, dl1_ceffs_0_10, dl1_beffs_0_10 = [], [], []
    dl1_leffs_0_10.append(leff_0_10)
    dl1_ceffs_0_10.append(ceff_0_10)
    dl1_beffs_0_10.append(beff_0_10)

    dl1_beffs_m_0_10 = np.ma.masked_equal(np.array(dl1_beffs_0_10),0)
    dl1_beffs_m_0_10.compressed()
    dl1_leffs_m_0_10 = np.ma.masked_equal(np.array(dl1_leffs_0_10),0)
    dl1_leffs_m_0_10.compressed()
    dl1_ceffs_m_0_10 = np.ma.masked_equal(np.array(dl1_ceffs_0_10),0)
    dl1_ceffs_m_0_10.compressed()

    print('leffs_0_10: ', dl1_leffs_m_0_10)    

    ######################0-10%
    plt.figure()
    plt.plot(dl1_beffs_m_0_10[0], 1. / dl1_leffs_m_0_10[0], color='C4', label='l-rej')
    #plt.plot(b_effs, 1./l_rej, color='C2', label='l-rej - pp recommendations')
    # plt.figure()
    plt.plot(dl1_beffs_0_10[0], 1. / dl1_ceffs_m_0_10[0],"--", color='C4', label='c-rej')
    #plt.plot(b_effs, 1./c_rej, "--", color='C2', label='c-rej - pp recommendations')
    
    plt.ylabel('background-rej')
    
    plt.legend()
    #plt.title('Default_0 0-10%')
    #plt.title('Default_0 50-80%')
    plt.title('Default_0 High $p_{t}$')
    #plt.title('Default_0 Low $p_{t}$')
    plt.yscale("log")
    plt.xlim(0.6,1)
    plt.ylim(0,3000)

    if (file == '_uns'):
        #ROC_b_0_10='pdf_files_uns/ROC_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        ROC_b_0_10='pdf_files_uns/ROC_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_uns/ROC_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_uns/ROC_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    else:
        #ROC_b_0_10='pdf_files_data/ROC_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        ROC_b_0_10='pdf_files_data/ROC_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_data/ROC_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
        #ROC_b_0_10='pdf_files_data/ROC_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.pdf'
    plt.savefig(ROC_b_0_10)

    plt.show()
 
    if (file == '_uns'):
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_b_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_uns/ROC_c_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    else:
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_b_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_c_vs_lrej_v_0_10_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_b_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_b_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        roc_c_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_c_vs_lrej_v_50_80_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_b_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_c_vs_lrej_v_highpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_b_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_b_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
        #roc_c_vs_l_txt_cent_0_10 = 'pdf_files_data/ROC_c_vs_lrej_v_lowpt_'+str(ver)+'_jz'+str(jz)+str(name)+'.txt'
    np.savetxt(roc_b_vs_l_txt_cent_0_10 , np.column_stack((dl1_beffs_0_10[0],1./np.array(dl1_leffs_0_10)[0])), delimiter=' ')
    np.savetxt(roc_c_vs_l_txt_cent_0_10 , np.column_stack((dl1_beffs_0_10[0],1./np.array(dl1_ceffs_0_10)[0])), delimiter=' ')
 """
    return   