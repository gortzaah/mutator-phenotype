#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:40:26 2023

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
from corrected_iSC_28Feb import * 
from average_data_dep import * 
from matplotlib.ticker import ScalarFormatter, NullFormatter
plt.rcParams.update({'font.size': 20})   
import itertools 


""" RUN THIS BEFORE RUNNING code_fig3c_graph.py """ 


def dynamics_last_time_point(mut_rate): 
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
    
    name_groups = ['time','pdiv','pdie', 'mut_impact', 'pos_muts', 'neg_muts', 'neutral_muts', 'id', 'space']
    name_str = "/Volumes/PhD_Data_MW/isc_data_dynamics/std=0.001/pdiv0=0.255/" + mut + "/mutations_"  
    name_str_last = ".csv"
    
    
    all_data_ = []   ## vector datas that should be merged 
    all_data_time = []
    all_data_rep = []
    all_data_ce = []
    all_data_cl = [] 
    #j = 5
    i = 0 
    for j in range(1,(no_rep+1)):
        name_str_new = name_str + "%d" %j + name_str_last
        dats = pd.read_csv(name_str_new,header=None,names=name_groups)
        
        
        
        #%% 
        # #1 group dats according to time and, if exist, find time of rapid population increase
        data_grp = dats.groupby(['time']).count()['pdiv'].reset_index()
        data_grp = data_grp.rename(columns = {'pdiv':'no_cells'})
        
        l_datas = len(data_grp)
        
        if (l_datas < 2):
            #print('meow')
            continue 
        
        else: 
            
            data_grp2 = dats.groupby(['time']) 
            last_group = (len(data_grp2.groups) - 1) * sample_frequency
            data_j = data_grp2.get_group(last_group)
            data_j = data_j.reset_index(drop = True)
            
            data_j['fitness'] = data_j['pdiv'] - data_j['pdie']
            data_j['1-fitness'] = 1 - data_j['fitness']
            data_j['pos_neg_muts'] = data_j['pos_muts'] - data_j['neg_muts']
            
            data_j['replicate'] = [j] * len(data_j)
            
            
            #all_data_rep.append([data_j['time'][0], j, len(data_j), len(np.unique(data_j['id']))])
            all_data_time.append(data_j['time'][0])
            all_data_rep.append(j)
            all_data_ce.append(len(data_j))
            all_data_cl.append(len(np.unique(data_j['id'])))
            
            data_j = data_j[['replicate', 'fitness', '1-fitness', 'pos_muts', 'neg_muts', 'pos_neg_muts']]
            
            all_data_.append(data_j)
            i = i + 1
        
    
    #%%
    #merge all data and create a big o data frame
    
    if (len(all_data_) > 0):  
    
        replicate_ = [all_data_[k]['replicate'] for k in range(len(all_data_))]
        replicate__ = list(itertools.chain.from_iterable(replicate_))    
            
        fitness_ = [all_data_[k]['fitness'] for k in range(len(all_data_))]
        fitness__ = list(itertools.chain.from_iterable(fitness_))
        
        ofitness_ = [all_data_[k]['1-fitness'] for k in range(len(all_data_))]
        ofitness__ = list(itertools.chain.from_iterable(ofitness_))
        
        pos_muts_ = [all_data_[k]['pos_muts'] for k in range(len(all_data_))]
        pos_muts__ = list(itertools.chain.from_iterable(pos_muts_))
        
        neg_muts_ = [all_data_[k]['neg_muts'] for k in range(len(all_data_))]
        neg_muts__ = list(itertools.chain.from_iterable(neg_muts_))
        
        pos_neg_muts_ = [all_data_[k]['pos_neg_muts'] for k in range(len(all_data_))]
        pos_neg_muts__ = list(itertools.chain.from_iterable(pos_neg_muts_))
            
        iSC_ = pd.DataFrame(list(zip(replicate__, fitness__, ofitness__, pos_muts__, neg_muts__, pos_neg_muts__)), columns = ['replicate','fitness','1-fitness', 'pos_muts', 'neg_muts', 'pos_neg_mutations'])
        
        iSC_rep = pd.DataFrame(list(zip(all_data_time, all_data_rep, all_data_ce, all_data_cl)), columns = ['time', 'replicate', 'no_cells', 'no_clones'])
    #%% 
    
    if 'iSC_' in locals():
            if (mut_rate == 1):
                iSC_0 = iSC_
                iSC_0.to_pickle("./iSC_0.pkl")  
                
                iSC_rep_0 = iSC_rep
                iSC_rep_0.to_pickle("./iSC_rep_0.pkl")  
                
            elif (mut_rate == 1e-1):
                iSC_1 = iSC_
                iSC_1.to_pickle("./iSC_1.pkl")
                
                iSC_rep_1 = iSC_rep
                iSC_rep_1.to_pickle("./iSC_rep_1.pkl")  
                
            elif (mut_rate == 1e-2):
                iSC_2 = iSC_
                iSC_2.to_pickle("./iSC_2.pkl") 
                
                iSC_rep_2 = iSC_rep
                iSC_rep_2.to_pickle("./iSC_rep_2.pkl") 
                
            elif (mut_rate == 1e-3):
                iSC_3 = iSC_
                iSC_3.to_pickle("./iSC_3.pkl") 
                
                iSC_rep_3 = iSC_rep
                iSC_rep_3.to_pickle("./iSC_rep_3.pkl") 
                
            elif (mut_rate == 1e-4):
                iSC_4 = iSC_
                iSC_4.to_pickle("./iSC_4.pkl") 
                
                iSC_rep_4 = iSC_rep
                iSC_rep_4.to_pickle("./iSC_rep_4.pkl") 
                
            elif (mut_rate == 0 ):
                iSC_null = iSC_
                iSC_null.to_pickle("./iSC_null.pkl") 
                
                iSC_rep_null = iSC_rep
                iSC_rep_null.to_pickle("./iSC_rep_null.pkl") 
            else:
                sys.exit("Meow meow mutations not existing meow meow!!") 
                
    


#%% 


dynamics_last_time_point(0)
dynamics_last_time_point(1)
dynamics_last_time_point(1e-1)
dynamics_last_time_point(1e-2)
dynamics_last_time_point(1e-3)
dynamics_last_time_point(1e-4)


#%% 



 
    