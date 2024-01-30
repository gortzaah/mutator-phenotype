"""
# Created by Malgorzata Tyczynska Weh in 2023 
# Script to demonstrate the evolutionary rescue potential of the mutator phenotype

### Important: while the x-axes of the figure reflect the deviation from the 
#zero net growth conditions (pdiv - pdie/(1-pdie)), the data is sorted after pdiv only
"""
#%% 
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib
import math
from scipy.stats import shapiro  
from average_data_dep import * 
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats
plt.rcParams.update({'font.size': 22})   
#%% 
#Define time 
time = np.linspace(0,40000,41) 
last_indx = 41 #<- corresponding to the last point from the the time series 


## Analyze how many replicates survive until the end of time;
# return: list of survival statistics
def analyze_survival(data, n, end_time):
    where_extinct = [np.count_nonzero(~np.isnan(data.iloc[i])) for i in range(n)]
    number_non_zero = np.sum([np.count_nonzero(~np.isnan(data.iloc[i])) == end_time for i in range(n)])
    percent_non_zero = (number_non_zero / n ) * 100
    which_non_zero = np.where([np.count_nonzero(~np.isnan(data.iloc[i])) == end_time for i in range(n)])[0]
    
    return [where_extinct, number_non_zero, percent_non_zero, which_non_zero]



### Function to analyze_survival to ALL the data: loop over the values of phenotypic
# variance (std_dev), values of probability of division (pdiv) and 
#mutation probabilities (mut_cases)


def get_survival_percentage(std_dev, i, j, last_indx):
    ## check which data to load 
    name_string_part1 = "/Volumes/PhD_Data_MW/data_evolutionary_rescue/" 
    
    if (std_dev == 0.001):
        name_string_part2 = std_strings[0] + "/"
    if (std_dev == 0.01):
        name_string_part2 = std_strings[1] + "/"
    if (std_dev == 0.1):
        name_string_part2 = std_strings[2] + "/"
    
    name_string_part3 = pdiv_strings[i] + "/" ### index i 
    name_string_part4 = mut_cases[j] + "/" ### index j
    name_string_part5 = "averages_"
    name_string_part6 = ".csv"
    
    string_total_first = name_string_part1 + name_string_part2 +  name_string_part3 + name_string_part4 + name_string_part5 
    data = get_trajectories(string_total_first,name_string_part6,1,no_simuls[j])
    data_no_cells = data[0]
    survival_data = analyze_survival(data_no_cells, no_simuls[j], last_indx)
    
    return survival_data[2]


#Apply get_survivial_percentage to analyze the data across mutation rates, pdivs 
#and phenotypic variances

std_numbers = [0.001, 0.01, 0.1]
std_strings = ['stddev=0.001','stddev=0.010','stddev=0.100']
pdiv_strings = ['pdiv=0.150','pdiv=0.200', 'pdiv=0.240','pdiv=0.250','pdiv=0.260','pdiv=0.300','pdiv=0.350']
mut_cases = ['both_0', 'both_1e-4', 'both_1e-3', 'both_1e-2', 'both_1e-1', 'both_1']
no_simuls = [250, 250, 250, 150, 75, 25] #in order: null, 1e-4, 1e-3, 1e-2, 1e-1, 1 /// INDEX J

#store the data 
ext_rates_001 = [] 
ext_rates_01 = [] 
ext_rates_1 = [] 

std_dev1 = std_numbers[0]

#Warning: this part takes some time to run, so make yourself a favor and grab a coffee! 
for i in range(len(pdiv_strings)):
    for j in range(len(mut_cases)):
        ext_rates_001.append(get_survival_percentage(std_numbers[0], i, j, last_indx ))
        ext_rates_01.append(get_survival_percentage(std_numbers[1], i, j, last_indx ))
        ext_rates_1.append(get_survival_percentage(std_numbers[2], i, j, last_indx ))

#reshape to ascending order
ext_rates_001p = np.reshape(ext_rates_001[::-1], (7,6)).T
ext_rates_01p = np.reshape(ext_rates_01[::-1], (7,6)).T
ext_rates_1p = np.reshape(ext_rates_1[::-1], (7,6)).T


#%% Part 1: for the example of how to read the heatmap 

dev = list(np.round(np.asarray([0.15,0.20, 0.24, 0.25, 0.26, 0.3, 0.35]) - 0.25,2))

plt.rcParams.update({'font.size': 28}) 

fig, ax = plt.subplots(1, sharex=True, sharey=True, figsize = (12,10)) 
fig.subplots_adjust(hspace= 0.2, wspace = 0.2)
im = ax.imshow(ext_rates_001p, vmin=0, vmax=100, aspect = 'auto')
cbar = ax.figure.colorbar(im, ax = ax)

ax.set_xticks(list(range(7)),dev[::-1], fontsize = 28)
#ax.set_xlabel('dev$_0$ \n deviation from \n \n  the zero net growth dynamics', labelpad = 20, fontsize = 36)
#ax.set_xlabel('predicted deviation from \n the zero net growth dynamics \n $p_{div}$ - $p_{die}$/(1-$p_{die}$)', labelpad = 20, fontsize = 36)
ax.set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1], fontsize = 28)
#ax.set_ylabel('p$_{mut}$ \n probability of mutation ', labelpad = 25, fontsize = 36)

plt.gca().invert_xaxis()
plt.savefig('./graphs/show_example_heatmap.pdf',bbox_inches='tight')

#%% Part 2: display survival percentage dependency on the phenotypic variance


plt.rcParams.update({'font.size': 22}) 

fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize = (28,8), gridspec_kw={'width_ratios': [1, 1, 1.25]}) 
fig.subplots_adjust(hspace= 0.2, wspace = 0.2)

ax[0].imshow(ext_rates_001p, vmin=0, vmax=100, aspect = 'auto')
ax[0].set_xticks(list(range(7)),dev[::-1], fontsize = 24)
ax[0].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1], fontsize = 28)
ax[0].set_ylabel('p$_{mut}$', labelpad = 25, fontsize = 36)
ax[0].set_title(' $\sigma$ = 1e-3', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        if (ext_rates_001p[i][j] >= 95):
            continue
        text = ax[0].text(j, i, round(ext_rates_001p[i][j], 1), ha = "center", va = "center", color = "w")


ax[1].imshow(ext_rates_01p, vmin=0, vmax=100, aspect = 'auto')
ax[1].set_xticks(list(range(7)),dev[::-1])
#ax[1].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[1].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
ax[1].set_ylabel('')
ax[1].set_title('$\sigma$ = 1e-2 ', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        if (ext_rates_01p[i][j] >= 95):
            continue
        text = ax[1].text(j, i, round(ext_rates_01p[i][j], 1), ha = "center", va = "center", color = "w")

im1 = ax[2].imshow(ext_rates_1p, vmin=0, vmax=100, aspect = 'auto' )
cbar = ax[2].figure.colorbar(im1, ax = ax[2], shrink = 1)
ax[2].set_xticks(list(range(7)),dev[::-1])
#ax[2].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[2].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
ax[2].set_ylabel('')
ax[2].set_title('$\sigma$ = 1e-1', fontsize = 36, pad = 20)
ax[2].autoscale()

plt.gca().invert_xaxis()

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(2)
    ax[1].spines[axis].set_linewidth(2)
    ax[2].spines[axis].set_linewidth(2)
ax[0].tick_params(width=2)
ax[1].tick_params(width=2)
ax[2].tick_params(width=2)

#display the percentages if they are below 95
for i in range(6):
    for j in range(7):
        if (ext_rates_1p[i][j] >= 95):
            continue
        text = ax[2].text(j, i, round(ext_rates_1p[i][j], 1), ha = "center", va = "center", color = "w")
fig.supxlabel('dev$_0$', y =-0.05, fontsize = 36)
#fig.supxlabel('$p_{div}$ - $p_{die}$/(1-$p_{die}$)', y =-0.05, fontsize = 36)
fig.suptitle('Survival percentage dependency \n on the phenotypic variance ($\sigma$)', fontsize = 48, y= 1.2) 

plt.savefig('./graphs/extinction_probability_heatmap_updt_12192023.pdf',bbox_inches='tight')


#%% Analyze the population sizes in the non-extinct replicates 

def analyze_survival_nocells(data, n, end_time):
    where_extinct = [np.count_nonzero(~np.isnan(data.iloc[i])) for i in range(n)]
    number_non_zero = np.sum([np.count_nonzero(~np.isnan(data.iloc[i])) == end_time for i in range(n)])
    percent_non_zero = (number_non_zero / n ) * 100
    which_non_zero = np.where([np.count_nonzero(~np.isnan(data.iloc[i])) == end_time for i in range(n)])[0]
    
    if (number_non_zero > 0): 
        pop_sizes = [data.iloc[which_non_zero[i]][last_indx-1] for i in range(len(which_non_zero))] 
        avg_size = np.sum(pop_sizes) / len(pop_sizes)
    else:
        pop_sizes = 0
        avg_size = 0
    
    return [where_extinct, number_non_zero, percent_non_zero, which_non_zero, pop_sizes, avg_size]

#apply to the data, similarly to "get_survival_percentage" 
def get_populations_survived(std_dev, i, j, last_indx):
    
    ## check which data to load 
    name_string_part1 = "/Volumes/PhD_Data_MW/data_evolutionary_rescue/" 
    
    if (std_dev == 0.001):
        name_string_part2 = std_strings[0] + "/"
    if (std_dev == 0.01):
        name_string_part2 = std_strings[1] + "/"
    if (std_dev == 0.1):
        name_string_part2 = std_strings[2] + "/"
    
    name_string_part3 = pdiv_strings[i] + "/" ### index i 
    name_string_part4 = mut_cases[j] + "/" ### index j
    name_string_part5 = "averages_"
    name_string_part6 = ".csv"
    
    string_total_first = name_string_part1 + name_string_part2 +  name_string_part3 + name_string_part4 + name_string_part5 

    data = get_trajectories(string_total_first,name_string_part6,1,no_simuls[j])

    data_no_cells = data[0]
    survival_data = analyze_survival_nocells(data_no_cells, no_simuls[j], last_indx)
    
    return survival_data[5]


## Apply for different mutation rates, phenotypic variances and pdiv values 
pop_sizes_001 = [] 
pop_sizes_01 = [] 
pop_sizes_1 = [] 

for i in range(len(pdiv_strings)):
    for j in range(len(mut_cases)):
        pop_sizes_001.append(get_populations_survived(std_numbers[0], i, j, last_indx ))
        pop_sizes_01.append(get_populations_survived(std_numbers[1], i, j, last_indx ))
        pop_sizes_1.append(get_populations_survived(std_numbers[2], i, j, last_indx ))

pop_sizes_001p = np.reshape(pop_sizes_001[::-1], (7,6)).T
pop_sizes_01p = np.reshape(pop_sizes_01[::-1], (7,6)).T
pop_sizes_1p = np.reshape(pop_sizes_1[::-1], (7,6)).T


#%% Plot

res_001 = np.ma.log(pop_sizes_001p)
res_01 = np.ma.log(pop_sizes_01p)
res_1 = np.ma.log(pop_sizes_1p)


dev = list(np.round(np.asarray([0.15,0.20, 0.24, 0.25, 0.26, 0.3, 0.35]) - 0.25,2))

K = 10000

fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize = (28,8), gridspec_kw={'width_ratios': [1, 1, 1.25]}) 
fig.subplots_adjust(hspace= 0.2, wspace = 0.2)
ax[0].imshow(pop_sizes_001p / 100, vmin=0, vmax=100, aspect = 'auto')
ax[0].set_xticks(list(range(7)),dev[::-1], fontsize = 24)
#ax[0].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[0].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1], fontsize = 28)
ax[0].set_ylabel('p$_{mut}$', labelpad = 25, fontsize = 36)
ax[0].set_title('$\sigma$ = 1e-3', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        prec = (pop_sizes_001p[i][j] / K) * 100
        if (prec >= 90):
            continue
        text = ax[0].text(j, i, round(prec, 1), ha = "center", va = "center", color = "w")
ax[1].imshow(pop_sizes_01p / 100, vmin=0, vmax=100, aspect = 'auto')
ax[1].set_xticks(list(range(7)),dev[::-1], fontsize = 24)
#ax[1].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[1].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
#ax[1].set_ylabel('p$_{mut}$', labelpad = 25, fontsize = 24)
ax[1].set_title('$\sigma$ = 1e-2', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        prec = (pop_sizes_01p[i][j] / K) * 100
        if (prec >= 90):
            continue
        text = ax[1].text(j, i, round(prec, 1), ha = "center", va = "center", color = "w")



im1 = ax[2].imshow(pop_sizes_1p / 100, vmin=0, vmax=100, aspect = 'auto' )
cbar = ax[2].figure.colorbar(im1, ax = ax[2], shrink = 1)
ax[2].set_xticks(list(range(7)),dev[::-1], fontsize = 24)
#ax[2].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[2].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
ax[2].set_ylabel('')
ax[2].set_title('$\sigma$ = 1e-1', fontsize = 36, pad = 20)
ax[2].autoscale()

fig.supxlabel('dev$_0$', y =-0.05, fontsize = 36)
fig.suptitle('Average population size dependency \n on the phenotypic variance ($\sigma$)', fontsize = 48, y= 1.2) 

plt.gca().invert_xaxis()

for i in range(6):
    for j in range(7):
        prec = (pop_sizes_1p[i][j] / K) * 100
        if (prec >= 90):
            continue
        text = ax[2].text(j, i, round(prec, 1), ha = "center", va = "center", color = "w")

plt.savefig('./graphs/extinction_final_popsize_heatmap_updt_12192023.pdf',bbox_inches='tight')       


#%% 



## Analyze how many replicates survive until the end of time;
# return: list of survival statistics
def analyze_fitness(data_fit, n, end_time):
    where_extinct = [np.count_nonzero(~np.isnan(data_fit.iloc[i])) for i in range(n)]
    number_non_zero = np.sum([np.count_nonzero(~np.isnan(data_fit.iloc[i])) == end_time for i in range(n)])
    percent_non_zero = (number_non_zero / n ) * 100
    which_non_zero = np.where([np.count_nonzero(~np.isnan(data_fit.iloc[i])) == end_time for i in range(n)])[0]

    if (number_non_zero > 0): 
        fit_all = [data_fit.iloc[which_non_zero[i]][last_indx-1] for i in range(len(which_non_zero))] 
        avg_fit = np.sum(fit_all) / number_non_zero#len(fit_all)
    else:
        fit_all = 0
        avg_fit = 0

    return [where_extinct, number_non_zero, percent_non_zero, which_non_zero, fit_all, avg_fit]



### Function to analyze_survival to ALL the data: loop over the values of phenotypic
# variance (std_dev), values of probability of division (pdiv) and 
#mutation probabilities (mut_cases)


def get_avg_fitness(std_dev, i, j, last_indx):
    ## check which data to load 
    name_string_part1 = "/Volumes/PhD_Data_MW/data_evolutionary_rescue/" 
    
    if (std_dev == 0.001):
        name_string_part2 = std_strings[0] + "/"
    if (std_dev == 0.01):
        name_string_part2 = std_strings[1] + "/"
    if (std_dev == 0.1):
        name_string_part2 = std_strings[2] + "/"
    
    name_string_part3 = pdiv_strings[i] + "/" ### index i 
    name_string_part4 = mut_cases[j] + "/" ### index j
    name_string_part5 = "averages_"
    name_string_part6 = ".csv"
    
    string_total_first = name_string_part1 + name_string_part2 +  name_string_part3 + name_string_part4 + name_string_part5 
    data = get_trajectories(string_total_first,name_string_part6,1,no_simuls[j])
    data_no_cells = data[0]
    
    data_no_cells = data[0]
    data_pdiv = data[1] / data_no_cells
    data_pdie = data[2] / data_no_cells
    data_fit = data_pdiv - data_pdie 

    fitness_data = analyze_fitness(data_fit, no_simuls[j], last_indx)
    
    return fitness_data[5]




#%% 
#Apply get_survivial_percentage to analyze the data across mutation rates, pdivs 
#and phenotypic variances




std_numbers = [0.001, 0.01, 0.1]
std_strings = ['stddev=0.001','stddev=0.010','stddev=0.100']
pdiv_strings = ['pdiv=0.150','pdiv=0.200', 'pdiv=0.240','pdiv=0.250','pdiv=0.260','pdiv=0.300','pdiv=0.350']
mut_cases = ['both_0', 'both_1e-4', 'both_1e-3', 'both_1e-2', 'both_1e-1', 'both_1']
no_simuls = [250, 250, 250, 150, 75, 25] #in order: null, 1e-4, 1e-3, 1e-2, 1e-1, 1 /// INDEX J

j = 0 # mutation case 
i = 0 # pdiv 

std_dev1 = std_numbers[0]

cat = get_avg_fitness(std_dev1, i, j, last_indx)

#%%

#store the data 
fit_001 = [] 
fit_01 = [] 
fit_1 = [] 


#Warning: this part takes some time to run, so make yourself a favor and grab a coffee! 
for i in range(len(pdiv_strings)):
    for j in range(len(mut_cases)):
        fit_001.append(get_avg_fitness(std_numbers[0], i, j, last_indx ))
        fit_01.append(get_avg_fitness(std_numbers[1], i, j, last_indx ))
        fit_1.append(get_avg_fitness(std_numbers[2], i, j, last_indx ))

#reshape to ascending order
fit_001p = np.reshape(fit_001[::-1], (7,6)).T
fit_01p = np.reshape(fit_01[::-1], (7,6)).T
fit_1p = np.reshape(fit_1[::-1], (7,6)).T

#%% 




plt.rcParams.update({'font.size': 22}) 

fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize = (28,8), gridspec_kw={'width_ratios': [1, 1, 1.25]}) 
fig.subplots_adjust(hspace= 0.2, wspace = 0.2)

ax[0].imshow(fit_001p, vmin=0.79, vmax=0.83, aspect = 'auto')
ax[0].set_xticks(list(range(7)),dev[::-1], fontsize = 24)
ax[0].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1], fontsize = 28)
ax[0].set_ylabel('p$_{mut}$', labelpad = 25, fontsize = 36)
ax[0].set_title(' $\sigma$ = 1e-3', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        if (fit_001p[i][j] >= 95):
            continue
        text = ax[0].text(j, i, round(fit_001p[i][j], 2), ha = "center", va = "center", color = "w")


ax[1].imshow(fit_01p, vmin=0.79, vmax=0.83, aspect = 'auto')
ax[1].set_xticks(list(range(7)),dev[::-1])
#ax[1].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[1].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
ax[1].set_ylabel('')
ax[1].set_title('$\sigma$ = 1e-2 ', fontsize = 36, pad = 20)
for i in range(6):
    for j in range(7):
        if (fit_01p[i][j] >= 95):
            continue
        text = ax[1].text(j, i, round(fit_01p[i][j], 2), ha = "center", va = "center", color = "w")

im1 = ax[2].imshow(fit_1p, vmin=0.79, vmax=0.83, aspect = 'auto' )
cbar = ax[2].figure.colorbar(im1, ax = ax[2], shrink = 1)
ax[2].set_xticks(list(range(7)),dev[::-1])
#ax[2].set_xlabel('mean p$_{div}$', labelpad = 15, fontsize = 24)
ax[2].set_yticks(list(range(6)), ['0', '1e-4', '1e-3', '1e-2', '1e-1', '1'][::-1])
ax[2].set_ylabel('')
ax[2].set_title('$\sigma$ = 1e-1', fontsize = 36, pad = 20)
ax[2].autoscale()

plt.gca().invert_xaxis()

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(2)
    ax[1].spines[axis].set_linewidth(2)
    ax[2].spines[axis].set_linewidth(2)
ax[0].tick_params(width=2)
ax[1].tick_params(width=2)
ax[2].tick_params(width=2)

#display the percentages if they are below 95
for i in range(6):
    for j in range(7):
        if (fit_1p[i][j] >= 95):
            continue
        text = ax[2].text(j, i, round(fit_1p[i][j], 2), ha = "center", va = "center", color = "w")

fig.supxlabel('dev$_0$', y =-0.05, fontsize = 36)
fig.suptitle('Fitness dependency \n on the phenotypic variance ($\sigma$)', fontsize = 48, y= 1.2) 

plt.savefig('./graphs/avg_fitness_heatmap_updt12202023.pdf',bbox_inches='tight')

#%% 


#see what you get if you just plot the values as lines, not heatmpas 


array_fit1 = np.sort(np.reshape(fit_1[::-1], [1,42]))
array_fit01 = np.sort(np.reshape(fit_01[::-1], [1,42]))
array_fit001 = np.sort(np.reshape(fit_001[::-1], [1,42]))


#%%
fig, ax = plt.subplots(1, figsize = (5,5))
ax.plot(array_fit1[0], '--', linewidth = 3, color = 'saddlebrown', label = '$\sigma = 0.1')
ax.plot(array_fit01[0], '-', linewidth = 3, color = 'tomato', label = '$\sigma = 0.01')
ax.plot(array_fit001[0], '-', linewidth = 3, color = 'darkorange', label = '$\sigma = 0.001')

ax.set_xlim(20,40)
ax.set_ylim(0.8,0.83)

