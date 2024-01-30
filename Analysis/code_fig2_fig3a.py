#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:13:50 2023

@author:  Malgorzata Tyczynska Weh
"""

#%%% INTIIALIZE 

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib
import math 
from average_data_dep import * 
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats
plt.rcParams.update({'font.size': 20})   


#%% Define time 
time = np.linspace(0,40000,401) 
last_indx = 401


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

#                                         READ THE DATA                                             #

#############################################################################################################


# For the classic data, the following no. of replicates is used as end_no_: 
    #pmut = null: 250
    #pmut = 1e-4: 250
    #pmut = 1e-3: 250
    #pmut = 1e-2: 150
    #pmut = 1e-1: 75
    #pmut = 1: 25


"""
p_mut = 0, p_back_mut = 0

all_dats, data_ --> both mutation rates are kept 
no_dep_all_dats, no_deps_data --> no div-dependent mut rate; only p_back_mut
no_indep_all_dats, no_indeps_data --> no div-independent mut rate; only p_mut

"""


start_no_null = 1 
end_no_null = 250

#
name_str_first_null = "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_0/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_0/averages_"  
name_str_last_null = ".csv"

all_dats_null = get_trajectories(name_str_first_null,name_str_last_null,start_no_null,end_no_null)
data_null_no_cells = all_dats_null[0]

data_null_pdiv = all_dats_null[1] / data_null_no_cells
data_null_pdie = all_dats_null[2] / data_null_no_cells
data_null_fit = data_null_pdiv - data_null_pdie

data_null_pos_muts = all_dats_null[3] / data_null_no_cells
data_null_neu_muts = all_dats_null[4] / data_null_no_cells
data_null_neg_muts = all_dats_null[5] / data_null_no_cells
data_null_diff_pos_muts = data_null_pos_muts - data_null_neg_muts
#%% 
"""
p_mut = 1e-4, p_back_mut = 1e-4

all_dats, data_ --> both mutation rates are kept 

"""


start_no_44 = 1 
end_no_44 = 250


name_str_first_44 = "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_1e-4/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_1e-4/averages_"  
name_str_last_44 = ".csv"

all_dats_44 = get_trajectories(name_str_first_44,name_str_last_44,start_no_44,end_no_44)
data_44_no_cells = all_dats_44[0]

data_44_pdiv = all_dats_44[1] / data_44_no_cells
data_44_pdie = all_dats_44[2] / data_44_no_cells
data_44_fit = data_44_pdiv - data_44_pdie

data_44_pos_muts = all_dats_44[3] / data_44_no_cells
data_44_neu_muts = all_dats_44[4] / data_44_no_cells
data_44_neg_muts = all_dats_44[5] / data_44_no_cells
data_44_diff_pos_muts = data_44_pos_muts - data_44_neg_muts


#%% 

"""
p_mut = 1e-3, p_back_mut = 1e-3

"""
start_no_33 = 1 
end_no_33 = 250

name_str_first_33 =  "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_1e-3/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_1e-3/averages_"  
name_str_last_33 = ".csv"


all_dats_33 = get_trajectories(name_str_first_33,name_str_last_33,start_no_33,end_no_33)
data_33_no_cells = all_dats_33[0]

data_33_pdiv = all_dats_33[1] / data_33_no_cells
data_33_pdie = all_dats_33[2] / data_33_no_cells
data_33_fit = data_33_pdiv - data_33_pdie

data_33_pos_muts = all_dats_33[3] / data_33_no_cells
data_33_neu_muts = all_dats_33[4] / data_33_no_cells
data_33_neg_muts = all_dats_33[5] / data_33_no_cells
data_33_diff_pos_muts = data_33_pos_muts - data_33_neg_muts

#%% 

"""
p_mut = 1e-2, p_back_mut = 1e-2

"""

start_no_22 = 1 
end_no_22 = 150


name_str_first_22 =  "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_1e-2/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_1e-2/averages_"  
name_str_last_22 = ".csv"


all_dats_22 = get_trajectories(name_str_first_22,name_str_last_22,start_no_22,end_no_22)

data_22_no_cells = all_dats_22[0]
data_22_pdiv = all_dats_22[1] / data_22_no_cells
data_22_pdie = all_dats_22[2] / data_22_no_cells
data_22_fit = data_22_pdiv - data_22_pdie

data_22_pos_muts = all_dats_22[3] / data_22_no_cells
data_22_neu_muts = all_dats_22[4] / data_22_no_cells
data_22_neg_muts = all_dats_22[5] / data_22_no_cells
data_22_diff_pos_muts = data_22_pos_muts - data_22_neg_muts

#%% 

"""
p_mut = 1e-1, p_back_mut = 1e-1

"""
start_no_11 = 1
end_no_11 = 75


name_str_first_11 =  "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_1e-1/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_1e-1/averages_"  
name_str_last_11 = ".csv"

all_dats_11 = get_trajectories(name_str_first_11,name_str_last_11,start_no_11,end_no_11)
data_11_no_cells = all_dats_11[0]
data_11_pdiv = all_dats_11[1] / data_11_no_cells
data_11_pdie = all_dats_11[2] / data_11_no_cells
data_11_fit = data_11_pdiv - data_11_pdie

data_11_pos_muts = all_dats_11[3] / data_11_no_cells
data_11_neu_muts = all_dats_11[4] / data_11_no_cells
data_11_neg_muts = all_dats_11[5] / data_11_no_cells
data_11_diff_pos_muts = data_11_pos_muts - data_11_neg_muts





#%% 

"""
p_mut = 1, p_back_mut = 1

"""
start_no_00 = 1
end_no_00 = 25

name_str_first_00 =  "/Volumes/PhD_Data_MW/redo_average_4January2024/std=0.001/pdiv0=0.255/both_1/averages_"
#"/Volumes/PhD_Data_MW/out_1March2023/classic/both_1/averages_"  
name_str_last_00 = ".csv"

# data_00 = get_stats(name_str_first_00,name_str_last_00,start_no_00,end_no_00)

all_dats_00 = get_trajectories(name_str_first_00,name_str_last_00,start_no_00,end_no_00)
data_00_no_cells = all_dats_00[0]

data_00_pdiv = all_dats_00[1] / data_00_no_cells
data_00_pdie = all_dats_00[2] / data_00_no_cells
data_00_fit = data_00_pdiv - data_00_pdie

data_00_pos_muts = all_dats_00[3] / data_00_no_cells
data_00_neu_muts = all_dats_00[4] / data_00_no_cells
data_00_neg_muts = all_dats_00[5] / data_00_no_cells
data_00_diff_pos_muts = data_00_pos_muts - data_00_neg_muts




#%% How many replicates survive? Pre-define the strings for the legend


survival_null = analyze_survival(data_null_no_cells, end_no_null, last_indx)
survival_44 = analyze_survival(data_44_no_cells, end_no_44, last_indx)
survival_33 = analyze_survival(data_33_no_cells, end_no_33, last_indx)
survival_22 = analyze_survival(data_22_no_cells, end_no_22, last_indx)
survival_11= analyze_survival(data_11_no_cells, end_no_11, last_indx)
survival_00= analyze_survival(data_00_no_cells, end_no_00, last_indx)


str_null = '$p_{mut}$ = 0, $\hat{n}$ = %d'  %survival_null[1]
str_44 = '$p_{mut}$ = 1e-4, $\hat{n}$ = %d'  %survival_44[1]
str_33 = '$p_{mut}$ = 1e-3, $\hat{n}$ = %d'  %survival_33[1]
str_22 = '$p_{mut}$ = 1e-2, $\hat{n}$ = %d'  %survival_22[1]
str_11 = '$p_{mut}$ = 1e-1, $\hat{n}$ = %d'  %survival_11[1]
str_00 = '$p_{mut}$ = 1, $\hat{n}$ = %d'  %survival_00[1]



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
                                                 MAIN FIGURE 2
"""
                                    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



"""     FIGURE 2A       """

no_cells_null = calculate_error(data_null_no_cells)
no_cells_44 = calculate_error(data_44_no_cells)
no_cells_33 = calculate_error(data_33_no_cells)
no_cells_22 = calculate_error(data_22_no_cells)
no_cells_11 = calculate_error(data_11_no_cells)
no_cells_00 = calculate_error(data_00_no_cells)

short = 151
# short corresponds to to time index of when to end the observation. 
#Max is 401 corresponding to time[401] = 40000 / (365 * 2) = 54.79 years
#Here we use 151 which is 20.68 years 


fig, ax = plt.subplots(1,figsize=(5,5))

ax.plot(time[0:short],no_cells_null[0][0:short], '--.', linewidth = 3,color='darkkhaki', label = str_null )
ax.fill_between(time[0:short], no_cells_null[1][0:short], no_cells_null[2][0:short][0:short], color = 'darkkhaki', alpha=0.3)

ax.plot(time[0:short],no_cells_44[0][0:short], '--.', linewidth = 3,color='darkred', label = str_44 )
ax.fill_between(time[0:short], no_cells_44[1][0:short], no_cells_44[2][0:short], color = 'red', alpha=0.3)

ax.plot(time[0:short],no_cells_33[0][0:short], '-.', linewidth = 3,color='darkgreen', label = str_33)
ax.fill_between(time[0:short], no_cells_33[1][0:short], no_cells_33[2][0:short], color = 'forestgreen', alpha=0.3)

ax.plot(time[0:short],no_cells_22[0][0:short], ':', linewidth = 3,color='darkslategray', label = str_22)
ax.fill_between(time[0:short], no_cells_22[1][0:short], no_cells_22[2][0:short], color = 'lightslategray', alpha=0.3)

ax.plot(time[0:short], no_cells_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
ax.fill_between(time[0:short], no_cells_11[1][0:short], no_cells_11[2][0:short], color = 'orange', alpha=0.4)

ax.plot(time[0:short], no_cells_00[0][0:short], '--', linewidth = 3,color='mediumblue', label = str_00)
ax.fill_between(time[0:short], no_cells_00[1][0:short], no_cells_00[2][0:short], color = 'royalblue', alpha=0.3)

ax.set_title('Average no. cells', pad=30)
ax.set_xlabel('time (years)')
ax.set_ylabel('No. cells')
ax.set_xticks(list(range(0,15000,730*5)), list(range(0,21,5)), fontsize = 20)

ax.set_yticks([0,2500,5000,7500,10000], [0,'','','',10000])


for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

#legend
#plt.legend(bbox_to_anchor=(1.07,0.9), borderaxespad=0,frameon=False, fontsize = 18) 

plt.savefig('./graphs/no_cells_v2_no_legend.pdf',bbox_inches='tight')

#%% 

"""     FIGURE 2B       """

fit_null = calculate_error(data_null_fit)
fit_44 = calculate_error(data_44_fit)
fit_33 = calculate_error(data_33_fit)
fit_22 = calculate_error(data_22_fit)
fit_11 = calculate_error(data_11_fit)
fit_00 = calculate_error(data_00_fit)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,5), sharex = True, gridspec_kw={'height_ratios': [6, 2]}) # gridspec_kw={'width_ratios': [5, 4]}
fig.subplots_adjust(hspace=0.25, left = 0.25)

short = 401

ax1.plot(time[0:short],fit_null[0][0:short], '--.', linewidth = 3,color='darkkhaki', label = str_null )
ax1.fill_between(time[0:short], fit_null[1][0:short], fit_null[2][0:short], color = 'darkkhaki', alpha=0.3)

ax1.plot(time[0:short],fit_44[0][0:short], '--.', linewidth = 3,color='darkred', label = str_44)
ax1.fill_between(time[0:short], fit_44[1][0:short], fit_44[2][0:short], color = 'red', alpha=0.3)

ax1.plot(time[0:short],fit_33[0][0:short], '-.', linewidth = 3,color='darkgreen', label = str_33)
ax1.fill_between(time[0:short], fit_33[1][0:short], fit_33[2][0:short], color = 'forestgreen', alpha=0.3)

ax1.plot(time,fit_22[0][0:short], ':', linewidth = 3,color='darkslategray', label = str_22)
ax1.fill_between(time[0:short], fit_22[1][0:short], fit_22[2][0:short], color = 'lightslategray', alpha=0.3)

ax1.plot(time[0:short], fit_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
ax1.fill_between(time[0:short], fit_11[1][0:short], fit_11[2][0:short], color = 'orange', alpha=0.4)

ax1.plot(time[0:short], fit_00[0][0:short], '--', linewidth = 3,color='mediumblue', label = str_00)
ax1.fill_between(time[0:short], fit_00[1][0:short], fit_00[2][0:short], color = 'royalblue', alpha=0.3)



ax2.plot(time[0:short],fit_null[0][0:short], '--.', linewidth = 3,color='darkkhaki', label = str_null )
ax2.fill_between(time[0:short], fit_null[1][0:short], fit_null[2][0:short], color = 'darkkhaki', alpha=0.3)

ax2.plot(time[0:short],fit_44[0][0:short], '--.', linewidth = 3,color='darkred', label = str_44 )
ax2.fill_between(time[0:short], fit_44[1][0:short], fit_44[2][0:short], color = 'red', alpha=0.3)

ax2.plot(time[0:short],fit_33[0][0:short], '-.', linewidth = 3,color='darkgreen', label = str_33)
ax2.fill_between(time[0:short], fit_33[1][0:short], fit_33[2][0:short], color = 'forestgreen', alpha=0.3)

ax2.plot(time,fit_22[0][0:short], ':', linewidth = 3,color='darkslategray', label = str_22)
ax2.fill_between(time[0:short], fit_22[1][0:short], fit_22[2][0:short], color = 'lightslategray', alpha=0.3)

ax2.plot(time[0:short], fit_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
ax2.fill_between(time[0:short], fit_11[1][0:short], fit_11[2][0:short], color = 'orange', alpha=0.4)

ax2.plot(time[0:short], fit_00[0][0:short], '--', linewidth = 3,color='mediumblue', label = str_00)
ax2.fill_between(time[0:short], fit_00[1][0:short], fit_00[2][0:short], color = 'royalblue', alpha=0.3)


# zoom-in / limit the view to different portions of the data
ax1.set_ylim(0.75, 0.81 )  # outliers only
ax2.set_ylim(0, 0.2)  # most of the data

ax1.set_yticks([0.75,0.8], fontsize = 20)
ax2.set_yticks([0,0.2], fontsize = 20)

#ax2.set_xticks(list(range(0,14601,730*5)), list(range(0,21,5)), fontsize = 18)
ax2.set_xticks(list(range(0,40151,730*25)), list(range(0,55,25)), fontsize = 20)
# ax1.set_xlim(0,11000)
# ax2.set_xlim(0,11000)
ax1.set_xlim(0,40151)
ax2.set_xlim(0,40151)
# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_xlabel('time (years)')
fig.supylabel('fitness', fontsize = 20)


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2)
    ax2.spines[axis].set_linewidth(2)
ax1.tick_params(width=2)
ax2.tick_params(width=2)

ax1.set_title('Average fitness', pad=30, fontsize = 20)

#plt.legend(bbox_to_anchor=(1.15,4.1), borderaxespad=0,frameon=False, fontsize = 18) 

plt.rcParams['figure.constrained_layout.use'] = False 

plt.savefig('./graphs/fitness_v2.pdf',bbox_inches='tight')


#%% 

"""     FIGURE 2C       """

###PART 1: all boxplots of average fitnesses 
fit_last = [ data_null_fit[last_indx-1][~np.isnan(data_null_fit[last_indx-1])],
            data_44_fit[last_indx-1][~np.isnan(data_44_fit[last_indx-1])],
            data_33_fit[last_indx-1][~np.isnan(data_33_fit[last_indx-1])],
            data_22_fit[last_indx-1][~np.isnan(data_22_fit[last_indx-1])],
            data_11_fit[last_indx-1][~np.isnan(data_11_fit[last_indx-1])],
            data_00_fit[last_indx-1][~np.isnan(data_00_fit[last_indx-1])]]

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6.5,5), sharex = True, gridspec_kw={'height_ratios': [8, 2]}) # gridspec_kw={'width_ratios': [5, 4]}
fig.subplots_adjust(hspace=0.25, left = 0.2, top =0.85, bottom = 0.01)

boxplt2 = ax1.boxplot(fit_last , showfliers=False , patch_artist=True) 
boxplt3 = ax2.boxplot(fit_last , showfliers=False , patch_artist=True) 


colors = list(reversed(['royalblue', 'orange', 'lightslategray', 'forestgreen', 'darkred', 'darkolivegreen']))


for patch, color in zip(boxplt2['boxes'], colors):
    patch.set_facecolor(color)
    #patch.set_alpha(0.75)

for patch, color in zip(boxplt3['boxes'], colors):
    patch.set_facecolor(color)
    #patch.set_alpha(0.75)


# # zoom-in / limit the view to different portions of the data
ax1.set_ylim(0.77, 0.81)  # outliers only
ax2.set_ylim(0, 0.2)  # most of the data


ax1.set_yticks([0.77,0.8]) # , fontsize = 18
ax2.set_yticks([0, 0.2]) #, fontsize = 18


ax2.set_xticks([1,2,3,4,5, 6], list(reversed(['1', '1e-1', '1e-2', '1e-3', '1e-4', '0'])))
ax2.set_xlabel('probability of mutation $p_{mut}$', labelpad = 10) #, fontsize = 18

# hide the spines between ax and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax2.set_xlabel('$p_{mut}$', fontsize = 20) #, fontsize = 18
fig.supylabel('fitness', fontsize = 20) #, fontsize = 18
fig.suptitle('Average fitness at t = t$_{end}$', fontsize = 24) #, fontsize = 20


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(2)
    ax2.spines[axis].set_linewidth(2)
ax1.tick_params(width=2)
ax2.tick_params(width=2)

plt.savefig('./graphs/fitness_boxplots1_v2.pdf',bbox_inches='tight')

#PART 2: zoom in 

fig, ax1 = plt.subplots(1,figsize=(6.5,5), sharex = True) # gridspec_kw={'width_ratios': [5, 4]}
fig.subplots_adjust(hspace=0.25, left = 0.2, top =0.85, bottom = 0.01)

boxplt2 = ax1.boxplot([fit_last[1], fit_last[2], fit_last[3], fit_last[4]] , showfliers=False , patch_artist=True ) 
color1 = ['darkred', 'forestgreen',  'lightslategray', 'orange' ]

for patch, color in zip(boxplt2['boxes'], color1):
    patch.set_facecolor(color)

ax1.set_xticks([1,2,3,4], list(['1e-4', '1e-3', '1e-2', '1e-1']))
ax1.set_xlabel('$p_{mut}$', labelpad = 10) #, fontsize = 18

ax1.set_yticks([0.801, 0.802, 0.803, 0.804])

ax1.set_ylabel('fitness', fontsize = 22) #, fontsize = 18
ax1.set_title('Average fitness at t = t$_{end}$\n', fontsize = 24)

plt.savefig('./graphs/fitness_boxplots2_v2.pdf',bbox_inches='tight')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
                                                 MAIN FIGURE 3A 
"""
                                    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

""" 
MAX CHANGE IN THE POPULATION SIZE

""" 
############## 


#nex = no extinction 
nex_data_null_no_cells = pd.DataFrame([data_null_no_cells.iloc[survival_null[2][i]] for i in range(survival_null[1])]).reset_index(drop = True)
nex_data_44_no_cells = pd.DataFrame([data_44_no_cells.iloc[survival_44[2][i]] for i in range(survival_44[1])]).reset_index(drop = True)
nex_data_33_no_cells = pd.DataFrame([data_33_no_cells.iloc[survival_33[2][i]] for i in range(survival_33[1])]).reset_index(drop = True)
nex_data_22_no_cells = pd.DataFrame([data_22_no_cells.iloc[survival_22[2][i]] for i in range(survival_22[1])]).reset_index(drop = True)
nex_data_11_no_cells = pd.DataFrame([data_11_no_cells.iloc[survival_11[2][i]] for i in range(survival_11[1])]).reset_index(drop = True)
nex_data_00_no_cells = pd.DataFrame([data_00_no_cells.iloc[survival_00[2][i]] for i in range(survival_00[1])]).reset_index(drop = True)

diff_no_cells_null_all = calculate_diff(nex_data_null_no_cells)
diff_no_cells_44_all = calculate_diff(nex_data_44_no_cells)
diff_no_cells_33_all = calculate_diff(nex_data_33_no_cells)
diff_no_cells_22_all = calculate_diff(nex_data_22_no_cells)
diff_no_cells_11_all = calculate_diff(nex_data_11_no_cells)
diff_no_cells_00_all = calculate_diff(nex_data_00_no_cells)


delta_max_null = np.asarray([time[diff_no_cells_null_all[1][i]] for i in range(len(diff_no_cells_null_all[1]))]).flatten()
delta_max_44 = np.asarray([time[diff_no_cells_44_all[1][i]] for i in range(len(diff_no_cells_44_all[1]))]).flatten()
delta_max_33 = np.asarray([time[diff_no_cells_33_all[1][i]] for i in range(len(diff_no_cells_33_all[1]))]).flatten()
delta_max_22 = np.asarray([time[diff_no_cells_22_all[1][i]] for i in range(len(diff_no_cells_22_all[1]))]).flatten()
delta_max_11 = np.asarray([time[diff_no_cells_11_all[1][i]] for i in range(len(diff_no_cells_11_all[1]))]).flatten()
delta_max_00 = np.asarray([time[diff_no_cells_00_all[1][i]] for i in range(len(diff_no_cells_00_all[1]))]).flatten()


"""
FIG. 3a: ILLUSTRATING THE CONCEPT OF DELTA
"""


fig, ax = plt.subplots(1,  sharey = True, figsize = (6,6)) 

idx_delta = diff_no_cells_11_all[1][15][0]
t_delta  = time[idx_delta]


for i in range(len(diff_no_cells_22_all[0])):
    ax.plot(time[0:short],data_11_no_cells.iloc[i][0:short], '-', linewidth = 1,color='darkorange' ) 


ax.hlines(y = data_11_no_cells.iloc[15][idx_delta], xmin = t_delta, xmax = (time[idx_delta+1]), linewidth = 2, color = 'black', linestyle = '--')
ax.vlines(x = (time[idx_delta+1]), ymin = data_11_no_cells.iloc[15][idx_delta], ymax = data_11_no_cells.iloc[15][idx_delta+1], linewidth = 2, color = 'black', linestyle = '--')

ax.plot(time[0:short], data_11_no_cells.iloc[15][0:short], '-', linewidth = 3, color='saddlebrown', label = 'sample \nreplicate')
ax.plot(time[0:short], data_11_no_cells.iloc[15][0:short], 'o', markersize = 12, color='saddlebrown', label = '')

ax.set_xticks([0, t_delta, 730], [0, '$\delta$', '1'])
ax.set_xlim(0,730)

ax.set_yticks([0,10000])

ax.set_xlabel('time (years)', labelpad = 15)
ax.set_ylabel('no. cells', labelpad =2)

plt.legend(bbox_to_anchor=(1.04,0.5), borderaxespad=0,frameon=False)

ax.set_title('Average number of cells', y = 1.07)

#plt.savefig('./graphs/delta_trajectories.pdf',bbox_inches='tight')




