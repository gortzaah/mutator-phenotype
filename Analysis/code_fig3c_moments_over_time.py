#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:10:32 2023

@author: Malgorzata Tyczynska Weh
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib
import math
from scipy.stats import shapiro  
#from functions_3February2023 import * 
from corrected_iSC_28Feb import * 
from average_data_dep import * 
from matplotlib.ticker import ScalarFormatter, NullFormatter
plt.rcParams.update({'font.size': 20})   

from scipy.stats import kurtosis
#


""" RUN THIS BEFORE RUNNING code_fig3c_graph.py """ 

## Function to calculate moments; we're using variance only but kurtosis could be fun too (to investigate the "fat-tailidness" of fitness distr.)


def calculate_fitness_moments(): 
    
    time_all = []
    mut_rate_all = []
    mean_all = []
    var_all = []
    skew_all = []
    kurt_all = []
    
    mutations = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    
    
    for k in range(len(mutations)):
        
        mut_rate = mutations[k]
    
        no_rep_null = 50
        no_rep_4 = 50 
        no_rep_3 = 35  #let's begin with a smaller number of experiments 
        no_rep_2 = 25
        no_rep_1 = 15
        no_rep_0 = 10
        
        if (mut_rate == 1):
            no_rep = no_rep_0
            mut = 'both_1'
        elif (mut_rate == 1e-1):
            no_rep = no_rep_1
            mut = 'both_1e-1'
        elif (mut_rate == 1e-2):
            no_rep = no_rep_2
            mut = 'both_1e-2'
        elif (mut_rate == 1e-3):
            no_rep = no_rep_3
            mut = 'both_1e-3'
        elif (mut_rate == 1e-4):
            no_rep = no_rep_4
            mut = 'both_1e-4'
        elif (mut_rate == 0 ):
            no_rep = no_rep_null
            mut = 'both_0'
        else:
            sys.exit("Meow meow mutations not existing meow meow!!") 
                
    
        end_time = 5500
        len_series = 221
        sample_frequency = 25
        K = 10000 #carrying capactiy   
        times = list(range(0,end_time+1,sample_frequency))
        
        #replicate index j 
        
        for j in range(1,no_rep+1):
            name_groups = ['time','pdiv','pdie', 'mut_impact', 'pos_muts', 'neg_muts', 'neutral_muts', 'id', 'space']
            name_str = "/Volumes/PhD_Data_MW/isc_data_dynamics/std=0.001/pdiv0=0.255/" + mut + "/mutations_"  
            name_str_last = ".csv"
            
            name_str_new = name_str + "%d" %j + name_str_last
            dats = pd.read_csv(name_str_new,header=None,names=name_groups)
            times_d = np.unique(dats['time'])
            
            dats['fitness'] = dats['pdiv'] - dats['pdie']
            #dats['mut_rate'] = [mut_rate] * len(dats)
            
            grp_dats = dats.groupby('time')
            
            
            for i in range(len(times_d)):
                dats_i = grp_dats.get_group(times_d[i]).reset_index(drop=True)
                time_all.append(times_d[i])   #want to keep tine and mutation rate informations 
                mut_rate_all.append(mut_rate)
                mean_all.append(dats_i['fitness'].mean())
                var_all.append(dats_i['fitness'].var())
                skew_all.append(dats_i['fitness'].skew())
                kurt_all.append(kurtosis(dats_i['fitness'], bias = False, nan_policy = 'omit' ))

    out_df = pd.DataFrame(list(zip(time_all,mut_rate_all, mean_all, var_all, skew_all, kurt_all)), columns = ['time', 'mut_rate', 'mean', 'var', 'skew', 'kurtosis'])

    return out_df 




out_moments = calculate_fitness_moments()


#%% 
out_moments.to_pickle("./out_moments.pkl")





