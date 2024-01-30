#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:10:08 2023

@author: Malgorzata Tyczynska Weh 
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib
import math
from matplotlib.ticker import ScalarFormatter, NullFormatter
import seaborn as sns 
import itertools 
plt.rcParams.update({'font.size': 28})


## Function to process the data around delta files to find individual cell fitness, 
#  & no. of mutations

def group_fit_pos_neg_muts(data_delta_):
    fitness_ = []
    pos_muts_ = []
    neg_muts_ = []
    total_no_cells_ = []
    time_deltas_ = []
    
    grp_delta_ = data_delta_.groupby(['delta_relative'])
    
    groups = list(range(-8,9,1)) 
    for i in groups:
        subgroup = grp_delta_.get_group(i)
        
        fit_list = list(itertools.chain.from_iterable(subgroup['fitness']))
        pos_muts_list = list(itertools.chain.from_iterable(subgroup['pos_muts']))
        neg_muts_list = list(itertools.chain.from_iterable(subgroup['neg_muts']))
        
        
        no_cells_all = sum(subgroup['no_cells'])
        fitness_.append(fit_list)
        pos_muts_.append(pos_muts_list)
        neg_muts_.append(neg_muts_list)
        total_no_cells_.append(no_cells_all) 
        time_deltas_.append([i] * no_cells_all)
        
    fitness_ = list(itertools.chain.from_iterable(fitness_))
    pos_muts_ = list(itertools.chain.from_iterable(pos_muts_))
    neg_muts_ = list(itertools.chain.from_iterable(neg_muts_))
    time_deltas_ = list(itertools.chain.from_iterable(time_deltas_))
    
    iSC_ = pd.DataFrame(list(zip(fitness_,pos_muts_, neg_muts_, time_deltas_)), columns = ['fitness', 'pos_muts', 'neg_muts', 'delta_relative']) 
    
    iSC_['diff_muts'] = iSC_['pos_muts'] - iSC_['neg_muts']
    
    return iSC_

#%% 

### Read the data_delta_X files, process with group_fit_pos_neg_muts and create the BIG plot around delta

data_delta_0 = pd.read_pickle("./data_delta_0.pkl")  
data_delta_1 = pd.read_pickle("./data_delta_1.pkl")  
data_delta_2 = pd.read_pickle("./data_delta_2.pkl")  
data_delta_3 = pd.read_pickle("./data_delta_3.pkl")  
data_delta_4 = pd.read_pickle("./data_delta_4.pkl")  


iSC_0 = group_fit_pos_neg_muts(data_delta_0)
iSC_1 = group_fit_pos_neg_muts(data_delta_1)
iSC_2 = group_fit_pos_neg_muts(data_delta_2)
iSC_3 = group_fit_pos_neg_muts(data_delta_3)
iSC_4 = group_fit_pos_neg_muts(data_delta_4)



### MAKE THE BIG O' FIGURE 


fig, ax = plt.subplots(3,5, figsize = (55,30),sharex = True) 

fig.subplots_adjust( hspace = 0.6, wspace = 0.5)

sns.stripplot(x = data_delta_4['delta_relative'] ,
      y = data_delta_4['no_cells'], ax = ax[0,0], size = 15, color = 'darkred')
    

sns.stripplot(x = data_delta_3['delta_relative'] ,
      y = data_delta_3['no_cells'], ax = ax[0,1], size = 15, color = 'forestgreen')
    
sns.stripplot(x = data_delta_2['delta_relative'] ,
          y = data_delta_2['no_cells'], ax = ax[0,2], size = 15, color = 'darkslategray')
    
    
sns.stripplot(x = data_delta_1['delta_relative'] ,
      y = data_delta_1['no_cells'], ax = ax[0,3], size = 15, color = 'darkorange')
    
sns.stripplot(x = data_delta_0['delta_relative'] ,
          y = data_delta_0['no_cells'], ax = ax[0,4], size = 15, color = 'darkblue')


str_4 = 'p$_{mut}$ = 1e-4' + ', $ \overline{n}$ = %d' %(len(data_delta_4) / 17)
str_3 = 'p$_{mut}$ = 1e-3' + ', $ \overline{n}$ = %d' %(len(data_delta_3) / 17)
str_2 = 'p$_{mut}$ = 1e-2' + ', $ \overline{n}$ = %d' %(len(data_delta_2) / 17)
str_1 = 'p$_{mut}$ = 1e-1' + ', $ \overline{n}$ = %d' %(len(data_delta_1) / 17)
str_0 = 'p$_{mut}$ = 1' + ', $ \overline{n}$ = %d' %(len(data_delta_0) / 17)

ax[0,0].set_title(str_4, y = 1.13, fontsize = 54)
ax[0,1].set_title(str_3, y = 1.13, fontsize = 54)
ax[0,2].set_title(str_2, y = 1.13, fontsize = 54)
ax[0,3].set_title(str_1, y = 1.13, fontsize = 54)
ax[0,4].set_title(str_0, y = 1.13, fontsize = 54)

plt.rcParams.update({'font.size': 60})



    #FITNESS
sns.stripplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['fitness'], size = 2, zorder = 0, ax=ax[1,0], color = 'darkred')

sns.boxplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['fitness'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,0], color = 'firebrick')


sns.stripplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['fitness'], size = 2, zorder = 0, ax=ax[1,1], color = 'forestgreen')

sns.boxplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['fitness'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,1], color = 'green')


sns.stripplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['fitness'], size = 2, zorder = 0, ax=ax[1,2], color = 'darkslategray')

sns.boxplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['fitness'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,2], color = 'slategray')


sns.stripplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['fitness'], size = 2, zorder = 0, ax=ax[1,3], color = 'darkorange')

sns.boxplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['fitness'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,3], color = 'orange')


sns.stripplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['fitness'], size = 2, zorder = 0, ax=ax[1,4], color = 'darkblue')

sns.boxplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['fitness'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,4], color = 'mediumblue')




## DIFFERENCE BETWEEN POSITIVE AND NEGATIVE MUTATIONS

    #FITNESS
sns.stripplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['pos_muts'], size = 2, zorder = 0, ax=ax[2,0], color = 'darkred')

sns.boxplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,0], color = 'firebrick')


sns.stripplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['pos_muts'], size = 2, zorder = 0, ax=ax[2,1], color = 'forestgreen')

sns.boxplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,1], color = 'green')


sns.stripplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['pos_muts'], size = 2, zorder = 0, ax=ax[2,2],  color = 'darkslategray')

sns.boxplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,2],  color = 'slategray')


sns.stripplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['pos_muts'], size = 2, zorder = 0, ax=ax[2,3], color = 'darkorange')

sns.boxplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,3], color = 'orange')


sns.stripplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['pos_muts'], size = 2, zorder = 0, ax=ax[2,4], color = 'darkblue')

sns.boxplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,4], color = 'mediumblue')




ax[0,0].set_xticks([0, 8, 16 ],['$\delta$ - 100', '$\delta$', '$\delta$ + 100'])




for i in range(5):
    ax[0,i].set_ylim(-500,10500)
    ax[1,i].set_ylim(-0.5,1)
    ax[2,i].set_ylim(-3,12)
    #if (i == 0):
    ax[0,i].set_yticks([0,2500, 5000, 7500, 10000],[0,'',5000,'',10000])
    ax[1,i].set_yticks([-0.5,0,0.5,1])
    ax[2,i].set_yticks([0, 5, 10])
    
    
    ax[0,i].set_ylabel('')
    ax[1,i].set_ylabel('')
    ax[2,i].set_ylabel('')
    
    ax[0,i].set_xlabel('')
    ax[1,i].set_xlabel('')
    ax[2,i].set_xlabel('')
    
    if (i > 0): 
        ax[0,i].set_yticks([0,2500, 5000, 7500, 10000], ['','','','',''])
        ax[1,i].set_yticks([-0.5,0,0.5,1],['','','',''])
        ax[2,i].set_yticks([0, 5, 10],['','',''])
        

fig.supxlabel('time relative to $\delta$ (days)', fontsize = 60) 
fig.suptitle('Cell dynamics during adaptation', y = 1, fontsize = 72)

str_4 = 'p$_{mut}$ = 1e-4' + ', $ \overline{n}$ = %d' %(len(data_delta_4) / 17)
str_3 = 'p$_{mut}$ = 1e-3' + ', $ \overline{n}$ = %d' %(len(data_delta_3) / 17)
str_2 = 'p$_{mut}$ = 1e-2' + ', $ \overline{n}$ = %d' %(len(data_delta_2) / 17)
str_1 = 'p$_{mut}$ = 1e-1' + ', $ \overline{n}$ = %d' %(len(data_delta_1) / 17)
str_0 = 'p$_{mut}$ = 1' + ', $ \overline{n}$ = %d' %(len(data_delta_0) / 17)




for j in range(5):
    for i in range(3):
        if (j ==0 ):
            ax[0,j].set_ylabel('no. cells', labelpad = 25, fontsize = 60)
            ax[1,j].set_ylabel('fitness', labelpad = 50, fontsize = 60)
            ax[2,j].set_ylabel('no. (+) mutations', labelpad = 80, fontsize = 60)
        else:
            ax[i,j].set_ylabel('')
            ax[i,j].tick_params('both', length=10, width=2, which='major')
            ax[i,j].set_xlabel('')
        
        if (i == 2):
            ax[i,j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],['$\delta$ - 100', '', '', '', '', '', '', '', '$\delta$', '', '', '', '','' , '', '', '$\delta$ + 100'], y =-0.08)
        else:
            ax[i,j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
            
#plt.savefig('/Users/80021045/Dropbox/PhD project/ABM_spring2023/graphs_adaptation_6April/cell_dynamics_during_adaptation.pdf',bbox_inches='tight')            
plt.savefig('./graphs/adaptation_dynamics.png',bbox_inches='tight')             



#%% 

## Supplementary

fig, ax =  plt.subplots(3,5, figsize = (55,30),sharex = True) 

fig.subplots_adjust( hspace = 0.6, wspace = 0.5)


str_4 = 'p$_{mut}$ = 1e-4' + ', $ \overline{n}$ = %d' %(len(data_delta_4) / 17)
str_3 = 'p$_{mut}$ = 1e-3' + ', $ \overline{n}$ = %d' %(len(data_delta_3) / 17)
str_2 = 'p$_{mut}$ = 1e-2' + ', $ \overline{n}$ = %d' %(len(data_delta_2) / 17)
str_1 = 'p$_{mut}$ = 1e-1' + ', $ \overline{n}$ = %d' %(len(data_delta_1) / 17)
str_0 = 'p$_{mut}$ = 1' + ', $ \overline{n}$ = %d' %(len(data_delta_0) / 17)

ax[0,0].set_title(str_4, y = 1.13, fontsize = 54)
ax[0,1].set_title(str_3, y = 1.13, fontsize = 54)
ax[0,2].set_title(str_2, y = 1.13, fontsize = 54)
ax[0,3].set_title(str_1, y = 1.13, fontsize = 54)
ax[0,4].set_title(str_0, y = 1.13, fontsize = 54)

plt.rcParams.update({'font.size': 60})


    #POSITIVE-NEGATIVE MUTATIONS
sns.stripplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['diff_muts'], size = 2, zorder = 0, ax=ax[0,0], color = 'darkred')

sns.boxplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['diff_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[0,0], color = 'firebrick')


sns.stripplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['diff_muts'], size = 2, zorder = 0, ax=ax[0,1], color = 'forestgreen')

sns.boxplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['diff_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[0,1], color = 'green')


sns.stripplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['diff_muts'], size = 2, zorder = 0, ax=ax[0,2],  color = 'darkslategray')

sns.boxplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['diff_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[0,2],  color = 'slategray')


sns.stripplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['diff_muts'], size = 2, zorder = 0, ax=ax[0,3], color = 'darkorange')

sns.boxplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['diff_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[0,3], color = 'orange')


sns.stripplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['diff_muts'], size = 2, zorder = 0, ax=ax[0,4], color = 'darkblue')

sns.boxplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['diff_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[0,4], color = 'mediumblue')




    #NEGATIVE MUTATIONS 
sns.stripplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['neg_muts'], size = 2, zorder = 0, ax=ax[1,0], color = 'darkred')

sns.boxplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['neg_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,0], color = 'firebrick')


sns.stripplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['neg_muts'], size = 2, zorder = 0, ax=ax[1,1], color = 'forestgreen')

sns.boxplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['neg_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,1], color = 'green')


sns.stripplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['neg_muts'], size = 2, zorder = 0, ax=ax[1,2],  color = 'darkslategray')

sns.boxplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['neg_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,2],  color = 'slategray')


sns.stripplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['neg_muts'], size = 2, zorder = 0, ax=ax[1,3], color = 'darkorange')

sns.boxplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['neg_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,3], color = 'orange')


sns.stripplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['neg_muts'], size = 2, zorder = 0, ax=ax[1,4], color = 'darkblue')

sns.boxplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['neg_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[1,4], color = 'mediumblue')

   #POSITIVE MUTATIONS FOR COMPARISON
sns.stripplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['pos_muts'], size = 2, zorder = 0, ax=ax[2,0], color = 'darkred')

sns.boxplot(x = iSC_4['delta_relative'] ,
      y = iSC_4['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,0], color = 'firebrick')


sns.stripplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['pos_muts'], size = 2, zorder = 0, ax=ax[2,1], color = 'forestgreen')

sns.boxplot(x = iSC_3['delta_relative'] ,
      y = iSC_3['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,1], color = 'green')


sns.stripplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['pos_muts'], size = 2, zorder = 0, ax=ax[2,2],  color = 'darkslategray')

sns.boxplot(x = iSC_2['delta_relative'] ,
      y = iSC_2['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,2],  color = 'slategray')


sns.stripplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['pos_muts'], size = 2, zorder = 0, ax=ax[2,3], color = 'darkorange')

sns.boxplot(x = iSC_1['delta_relative'] ,
      y = iSC_1['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,3], color = 'orange')


sns.stripplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['pos_muts'], size = 2, zorder = 0, ax=ax[2,4], color = 'darkblue')

sns.boxplot(x = iSC_0['delta_relative'] ,
      y = iSC_0['pos_muts'], fliersize = 0, zorder = 1, linewidth=3, ax = ax[2,4], color = 'mediumblue')




for i in range(5):
    ax[0,i].set_ylim(-3,12)
    ax[1,i].set_ylim(-3,12)
    ax[2,i].set_ylim(-3,12)
 
    #if (i == 0):
    ax[0,i].set_yticks([0, 5, 10])
    ax[1,i].set_yticks([0, 5, 10])
    ax[2,i].set_yticks([0, 5, 10])
    
    ax[0,i].set_ylabel('')
    ax[1,i].set_ylabel('')
    ax[2,i].set_ylabel('')
    
    ax[0,i].set_xlabel('')
    ax[1,i].set_xlabel('')
    ax[2,i].set_xlabel('')
    
    if (i > 0): 
        ax[0,i].set_yticks([ 0, 5, 10],['','',''])
        ax[1,i].set_yticks([ 0, 5, 10],['','',''])
        ax[2,i].set_yticks([ 0, 5, 10],['','',''])


for j in range(5):
    for i in range(3):
        if (j ==0 ):
            ax[0,j].set_ylabel('no. (+) - no. (-) \n mutations', labelpad = 25, fontsize = 60)
            ax[1,j].set_ylabel('no. (-) mutations', labelpad = 50, fontsize = 60)
            ax[2,j].set_ylabel('no. (+) mutations', labelpad = 50, fontsize = 60)
        else:
            ax[i,j].set_ylabel('')
            ax[i,j].tick_params('both', length=10, width=2, which='major')
            ax[i,j].set_xlabel('')
        
        if (i == 2):
            ax[i,j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],['$\delta$ - 100', '', '', '', '', '', '', '', '$\delta$', '', '', '', '','' , '', '', '$\delta$ + 100'], y =-0.08)
        else:
            ax[i,j].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ],['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
            

fig.supxlabel('time relative to $\delta$ (days)', fontsize = 60, y = -0.01) 
fig.suptitle('Cell dynamics during adaptation', y = 1.02, fontsize = 72)

plt.savefig('./graphs/supplementary_adaptation_dynamics.png',bbox_inches='tight')            
