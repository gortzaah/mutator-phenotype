#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:46:14 2023

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

import seaborn as sns 


#%% from code_fig3c_moments_over_time.py
out_moments = pd.read_pickle("./out_moments.pkl")

#%%
data_out = out_moments.groupby(['mut_rate'])


replicates = [50, 50, 35, 25, 15, 10]
mutations = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
mutations_title = ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1']

colors = ['olive', 'darkred', 'forestgreen', 'darkslategray', 'darkorange', 'royalblue']


#%% 


## This comes from code_fig3c_boxplot_end_variance.py
iSC_null = pd.read_pickle("./iSC_null.pkl")  
iSC_0 = pd.read_pickle("./iSC_0.pkl")  
iSC_1 = pd.read_pickle("./iSC_1.pkl")  
iSC_2 = pd.read_pickle("./iSC_2.pkl")  
iSC_3 = pd.read_pickle("./iSC_3.pkl")  
iSC_4 = pd.read_pickle("./iSC_4.pkl")  





#%% 


mutations = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
mutations_title = ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1']


def get_last_time_fitness_var(mut_rate):
    if (mut_rate == 0): 
        iSC_ = iSC_null
    elif(mut_rate == 1e-4):
        iSC_ = iSC_4
    elif(mut_rate == 1e-3):
        iSC_ = iSC_3    
    elif(mut_rate == 1e-2):
        iSC_ = iSC_2    
    elif(mut_rate == 1e-1):
        iSC_ = iSC_1    
    elif(mut_rate == 1e-0):
        iSC_ = iSC_0    
        
    
    grp_ = iSC_.groupby('replicate')
    grp__len = list(grp_.groups.keys())
    vars_ = []
    for i in grp__len:
        
        iSC__rep = grp_.get_group(i).reset_index(drop=True)
        vars_.append(iSC__rep['fitness'].var())
    
    out = pd.DataFrame(vars_, columns = ['var'])
    
    return out 

list_iSC_pd = []
for j in mutations:
    list_iSC_pd.append(get_last_time_fitness_var(j))




#%% 


def get_var_stats(mut_rate_):
    data_ = data_out.get_group(mut_rate_)
    
    q25_list = []
    q50_list = []
    q75_list = []
    mean_list = []
    
    
    
    #for each time point 
    times_list = data_['time'].unique()
    for t in times_list:
    
        data_temp_ = data_['var'][data_['time'] == t]
        
        quants = np.quantile(data_temp_,[0.25,0.5,0.75])
        q25_list.append(quants[0])
        q50_list.append(quants[1])
        q75_list.append(quants[2])
        mean_list.append(np.mean(data_temp_))
    
    
    
    data_stats_ = pd.DataFrame(list(zip(times_list, q25_list, q50_list, q75_list, mean_list )), columns = ['time', 'q25', 'q50', 'q75', 'mean'])                

    return data_stats_



    


fig, ax = plt.subplots(1,2, figsize = (15,5), sharey=(True), gridspec_kw={'width_ratios': [10, 1]})
fig.subplots_adjust(wspace = 0.05)

c_null = (0.33, 0.42, 0.18, 0.5) 
c_4 = (.54, 0, 0, .5)
c_3 = (0.13, 0.54, 0.13, 0.4)
c_2 = (0.18, 0.3, 0.3, 0.6)
c_1 = (1, 0.54, 0, 0.5)
c_0 = (0.1, 0.25, 0.67, 0.5)

cl_null = (0.33, 0.42, 0.18, 1) 
cl_4 = (.54, 0, 0, 1)
cl_3 = (0.13, 0.54, 0.13, 1)
cl_2 = (0.18, 0.3, 0.3, 1)
cl_1 = (1, 0.54, 0, 1)
cl_0 = (0.1, 0.25, 0.67, 1)


colors_shade = [c_null, c_4, c_3, c_2, c_1, c_0 ]    
colors_line = [c_null, cl_4, cl_3, cl_2, cl_1, cl_0 ]    

for i, j in enumerate(mutations):
    
    pd_stats = get_var_stats(j)
    ax[0].plot(pd_stats['time'], pd_stats['q50'], color = colors_line[i], linewidth = 3)
    ax[0].fill_between(pd_stats['time'], pd_stats['q25'], pd_stats['q75'], color = colors_shade[i])
  
    
ax[0].set_yscale('log')
ax[0].set_ylim(1e-6,1)
ax[0].set_xticks([0, 2.5*730, 5*730, 7.5*730], [0, 2.5, 5, 7.5])
ax[0].set_xlabel('')
ax[0].set_ylabel('variance (log$_{10}$)', labelpad = 15, fontsize = 24)
#ax[1].set_box_aspect(2/1)


for i in range(len(mutations)):
    #data_ = data_out.get_group(mutations[i])
    data_l = list_iSC_pd[i][list_iSC_pd[i]>1e-32]
    
    sns.boxplot(y = data_l['var'],ax = ax[1], boxprops=dict(facecolor=colors_shade[i], color=colors_shade[i]),
                 capprops=dict(color=colors_line[i]),
                 whiskerprops=dict(color=colors_line[i]),
                flierprops=dict(color=colors_shade[i], markeredgecolor=colors_line[i]),
                medianprops=dict(color=colors_shade[i]), fliersize = 0)
          
ax[1].set_ylabel('')           
ax[1].set_xticks([0], [20.5])

fig.supxlabel('time (years)', y = -0.05)
fig.suptitle('Variance of fitness over time', y = 1.02, fontsize = 28)

plt.savefig('./graphs/variance_fitness.pdf',bbox_inches='tight')
# plt.savefig('./variance_fitness.png',bbox_inches='tight')



