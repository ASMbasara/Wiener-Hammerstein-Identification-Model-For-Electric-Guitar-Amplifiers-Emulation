# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:22:26 2023

@author: jhvaz
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_size = 10
y_size = 6

def simple_plot(x, y1, title, x_label, y_label, x_scale=1, y_scale=1, x_axis='lin'):
    plt.figure(figsize=(x_size, y_size))
    plt.plot(x, y1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def double_plot(x1, y1, x2, y2, title1, x_label, y_label1, title2, y_label2, x_scale=1, y_scale=1, x_axis='lin'):
    plt.figure(figsize=(x_size, y_size))
    
    plt.subplot(2, 1, 1)
    if x_axis=='log':
        plt.semilogx(x1, y1)
    else:
        plt.plot(x1,y1)
            
    plt.title(title1)
    plt.xlabel(x_label)
    plt.ylabel(y_label1)
    plt.grid()

    plt.subplot(2, 1, 2)
    if x_axis=='log':
        plt.semilogx(x2, y2)
    else:
        plt.plot(x2,y2)
    plt.title(title2)
    plt.xlabel(x_label)
    plt.ylabel(y_label2)
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def plot_comparison(fig_num, x, y1, y2, title, x_label, y_label, x_legend=" ", y_legend=" ", x_scale=1, y_scale=1, x_axis='lin'):
    plt.figure(fig_num,figsize=(x_size, y_size))
    plt.clf()
    plt.plot(x, y1, label=x_legend)
    plt.plot(x, y2, label=y_legend)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.grid()
    
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)
    plt.pause(0.001)
    
    