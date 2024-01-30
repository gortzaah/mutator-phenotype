#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:10:08 2023

@author: Malgorzata Tyczynska Weh 
"""


import os, sys
import numpy as np
import pandas as pd
import csv
import math


"""
Code to generate figure 3b: part 1

This code executes a function dynamics_around_delta which:
    1. for each replicate, reads iSC data for  a user-specified mutation rate 
    (1: pmut = 1; 1e-1: pmut = 1e-1 etc.)
    2. for each replicate, finds the time and index of the maximal population increase (delta), 
    and the 7 observations prior and after delta (7 time steps = 87.5 days)
    3. for each repli





"""




def dynamics_around_delta(mut_rate): 
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
    K = 10000 #carrying capactiy   
    
    
    
    name_groups = ['time','pdiv','pdie', 'mut_impact', 'pos_muts', 'neg_muts', 'neutral_muts', 'id', 'space']
    name_str = "/Volumes/PhD_Data_MW/isc_data_dynamics/std=0.001/pdiv0=0.255/" + mut + "/mutations_"  
    name_str_last = ".csv"
    
    
    all_data_delta = []
    #j = 5 
    for j in range(1,(no_rep+1)):
        name_str_new = name_str + "%d" %j + name_str_last
        dats = pd.read_csv(name_str_new,header=None,names=name_groups)
        
        ### Regroup the data so you sample every 25th step 
        
        #%% 
        #1 group dats according to time and, if exist, find time of rapid population increase
        data_grp = dats.groupby(['time']).count()['pdiv'].reset_index()
        data_grp = data_grp.rename(columns = {'pdiv':'no_cells'})
        
        #Create delta and epslion 
            # delta: there is a rapid population increase 
            # epsilon: there are only small, stochastic fluctuations
        
        #analyze only the replicates that do not go extinct
        if (data_grp.index[-1] < (len_series-1)):
            continue
        
        if (data_grp['no_cells'].iloc[len_series - 1] >= (K/2) ):
            #if true, this is delta
            
            #find dN/dt: the change in the number of cells over time
            dN = list(np.asarray(data_grp['no_cells'].iloc[1:len_series]) - np.asarray(data_grp['no_cells'].iloc[0:(len_series - 1)]))
            dN.append(0)
            
            #find the max change and the time
            N_delta = np.max(dN)
            delta = np.where(dN == np.max(dN))[0][0]
            
            dN = pd.Series(dN)
            dN.name = 'dN'
            
            index_delta = list(range(delta - 8, delta + 9, 1))  ## also changed here 
            l_delta_idx = len(index_delta)
            
            delta_relative = pd.Series(index_delta - delta)
            delta_relative.name = 'delta_relative'
            
                #if some indexes are negative, find the 0th index and repeat the information the required no of times
            if (any( (i < 0 ) for i in index_delta)):
                first_indx = np.where(np.asarray(index_delta) == 0)[0][0]  # prev: == 
                index_delta[0:first_indx] = [0] * first_indx
            
                #if some indexes exceed length of series, repeat the information at the last datapoint 
            if (any ( (i > (len_series - 1)) for i in index_delta )): 
                first_indx = np.where(np.asarray(index_delta) == (len_series - 1))[0][-1] # prev: ==
                index_delta[first_indx : l_delta_idx] = [(len_series - 1)] * (l_delta_idx - first_indx)    
            
    
            data_delta = data_grp.iloc[index_delta]
            data_delta = data_delta.reset_index(drop=True)
            data_delta['delta_relative'] = delta_relative
            data_delta['replicate'] = [j] * l_delta_idx    
            data_delta['dN'] = list(dN[index_delta])
           
         
            data_delta['fitness'] = [[]] * l_delta_idx
            data_delta['pos_muts'] = [[]] * l_delta_idx
            data_delta['neg_muts'] = [[]] * l_delta_idx
            data_delta['id_count'] = [0] * l_delta_idx
        
            #find indexes of 
            for k in range(l_delta_idx):
                all_datas_time_k = list(np.where(dats['time'] == data_delta['time'][k])[0])
                all_datas_k = dats.iloc[all_datas_time_k]
                fitness_k = all_datas_k['pdiv'] - all_datas_k['pdie']
                data_delta['fitness'][k] = list(fitness_k)
                data_delta['pos_muts'][k] = list(all_datas_k['pos_muts'])
                data_delta['neg_muts'][k] = list(all_datas_k['neg_muts'])
                data_delta['id_count'][k] = len(np.unique(dats.iloc[all_datas_time_k]['id']))
        
        
        if 'data_delta' in locals():
            all_data_delta.append(data_delta)    
      
        
                
    if len(all_data_delta) != 0:            
        data_out = pd.concat(all_data_delta) 
        
        if (mut_rate == 1):
            data_delta_0 = data_out
            data_delta_0.to_pickle("./data_delta_0.pkl")  
        elif (mut_rate == 1e-1):
            data_delta_1 = data_out
            data_delta_1.to_pickle("./data_delta_1.pkl") 
        elif (mut_rate == 1e-2):
            data_delta_2 = data_out
            data_delta_2.to_pickle("./data_delta_2.pkl") 
        elif (mut_rate == 1e-3):
            data_delta_3 = data_out
            data_delta_3.to_pickle("./data_delta_3.pkl") 
        elif (mut_rate == 1e-4):
            data_delta_4 = data_out
            data_delta_4.to_pickle("./data_delta_4.pkl") 
        elif (mut_rate == 0 ):
            data_delta_null = data_out
            data_delta_null.to_pickle("./data_delta_null.pkl") 
        else:
            sys.exit("Meow meow mutations not existing meow meow!!") 
            
            
    
#%%     

dynamics_around_delta(1)
dynamics_around_delta(1e-1)
dynamics_around_delta(1e-2)
dynamics_around_delta(1e-3)
dynamics_around_delta(1e-4)
dynamics_around_delta(0)


























#%% for each 