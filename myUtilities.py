# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:13:36 2023

@author: jhvaz
"""
import numpy as np
import pandas as pd

def replace_char(string, char, replacement):
    res = ''
    for c in string:
        if c==char:
            c=replacement
        res+=c
    return res        


def db2Lin(num):
    return 10**(num/20)

def lin2DB(num):
    return 20*np.log10(num)

def read_simulation_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_data = []
    lines[0]=lines[0].replace('\n','')
    columns = lines[0].split('\t')
    df = None

    for line in lines[1:]:
        # Read data lines and append to the current data list
        values = line.split()
        current_data.append([float(values[0]), float(values[1]), float(values[2])])

    if current_data:
        df = pd.DataFrame(current_data, columns=columns)


    return df