# -*- coding: utf-8 -*-
"""
plot bar plots for the PD cross-lingual paper
@author: yiting
"""
# comment/uncomment plt.legend part before running
# running this file will create png and jpg files in the current folder
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
def plot_langs(values1_f1, values2_f1, values1_auc, values2_auc, titles, barWidth, size, filenames):
    
    # Set position of bar on X axis
    br1 = np.arange(len(values1_f1[0]))
    br2 = [x + barWidth for x in br1]
     
    # Make the plot
    for i in range(len(values1_f1)):
        
        # F1
        fig = plt.subplots(figsize =size)
        plt.ylim([60,100])
        plt.bar(br1, values1_f1[i], color ='firebrick', width = barWidth,
                edgecolor ='grey', label ='Interpretable')
        plt.bar(br2, values2_f1[i], color ='steelblue', width = barWidth,
                edgecolor ='grey', label ='Non-interpretable')
        # Adding Xticks
        plt.xlabel(titles[i], fontweight ='bold', fontsize = 42)
        plt.ylabel('Max F1-score (%)', fontweight ='bold', fontsize = 40)
        plt.xticks([r + barWidth for r in range(len(values1_f1[0]))],
                ['mono', 'multi', 'cross'],fontsize=40)
        plt.yticks(fontsize=40)
        plt.legend(fontsize=38,loc='upper left')
        plt.savefig(filenames[i]+'_f1', dpi=300, bbox_inches = 'tight')
        Image.open(filenames[i]+'_f1.png').convert('RGB').save(filenames[i]+'_f1.jpg','JPEG',dpi=(300, 300))
        
        # AUC
        fig = plt.subplots(figsize =size)
        plt.ylim([0.60,1.00])
        plt.bar(br1, values1_auc[i], color ='firebrick', width = barWidth,
                edgecolor ='grey', label ='Interpretable')
        plt.bar(br2, values2_auc[i], color ='steelblue', width = barWidth,
                edgecolor ='grey', label ='Non-interpretable')
        # Adding Xticks
        plt.xlabel(titles[i], fontweight ='bold', fontsize = 42)
        plt.ylabel('Max AUC', fontweight ='bold', fontsize = 40)
        plt.xticks([r + barWidth for r in range(len(values1_auc[0]))],
                ['mono', 'multi', 'cross'],fontsize=40)
        plt.yticks(fontsize=40)
        # plt.legend(fontsize=38,loc='upper left')
        plt.savefig(filenames[i]+'_auc', dpi=300, bbox_inches = 'tight')
        Image.open(filenames[i]+'_auc.png').convert('RGB').save(filenames[i]+'_auc.jpg','JPEG',dpi=(300, 300))
        
# set width of bar
barWidth = 0.25
size = (12,10)

# bar values - F1
# mono, multi, cross
interp_f1 = [[73, 82, 76],
             [90, 86, 73],
             [76, 76, 66],
             [80, 82, 70],
             [72, 71, 66],
             [98, 87, 77]]
# # 7th layer wav2vec2 and hubert
# non_interp_f1 = [[78, 85, 71], # NLS cross 68->71
#                  [90, 84, 77], # Neurovoz cross 71->77
#                  [80, 85, 74],
#                  [84, 91, 89], # czech mono 83->84
#                  [84, 81, 82],
#                  [89, 95, 79]] # Ita multi 90->95
# 4th layer wav2vec2 and 7th layer hubert
non_interp_f1 = [[78, 85, 72], # nls cross 68->72
                 [91, 91, 77], # neu 90->91, 84->91, 71->77
                 [80, 85, 74],
                 [86, 91, 89], # czech 83->86
                 [86, 82, 82], # gita 84->86, multi 81->82
                 [89, 95, 79]] # ita multi 90->95
 # bar values - AUC
interp_auc = [[0.82, 0.83, 0.79],
              [0.95, 0.92, 0.85],
              [0.84, 0.83, 0.64],
              [0.85, 0.85, 0.79],
              [0.77, 0.78, 0.66],
              [1.00, 0.93, 0.84]]
# # # 7th layer wav2vec2 and hubert
# non_interp_auc = [[0.82, 0.92, 0.89],
#                   [0.96, 0.93, 0.90], # Neurovoz mono 95->96
#                   [0.88, 0.88, 0.84], # Ger mono 84->88
#                   [0.92, 0.93, 0.97],
#                   [0.89, 0.91, 0.89], # Gita mono 87->89
#                   [1.00, 0.99, 0.94]] # Ita multi 97->99
# 4th layer wav2vec2 and 7th layer hubert
non_interp_auc = [[0.82, 0.92, 0.89], #
                  [0.96, 0.96, 0.90], # neu mono 95->96, multi 93->96
                  [0.88, 0.88, 0.84], # ger mono 84->88
                  [0.92, 0.93, 0.97],
                  [0.90, 0.91, 0.89], # gita mono 87->90
                  [1.00, 0.99, 0.94]] # ita multi 97->99

titles = ['NLS','Neurovoz','GermanPD','CzechPD','GITA','ItalianPVS']
filenames = ['nls','neurovoz','german','czech','gita','italian']

plot_langs(values1_f1=interp_f1, values2_f1=non_interp_f1, \
           values1_auc=interp_auc, values2_auc=non_interp_auc, titles=titles, \
               barWidth=barWidth, size=size, filenames=filenames)


