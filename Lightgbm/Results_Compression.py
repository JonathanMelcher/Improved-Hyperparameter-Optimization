import numpy as np
import pandas as pd
import os
import sys
import pickle


### This module is used to read in the data from the LGBM analysis run on 
### the cluster. The goal is to correctly order the 256 individual runs
### in a data file to be send to Michal for further analysis.

base_name = ['true_hyper', 'grid_rand', 'hyp', 'lhs', 'ran', 'reg','rsa']
dimension = np.array([2,3,4])
resolution = np.array([2,3,4,5])
batches = np.arange(0,32,1)


data_path = 'data_LGBM/'
file_names = os.listdir(data_path)
file_names.sort()



# find all files starting with true_hyper_d_3_res_5 in the file names list
def combine_data_rank_0(BASE_NAME, DIM,RES, save = False):
    big_array = np.zeros((8*31, DIM+1))
    number_of_points = np.zeros(8*31)
    temp_store_values = np.zeros((8, DIM+1))
    normal_files = []
    for file_name in file_names:
        if file_name.startswith(BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES)):
            if not file_name.endswith('_rank_0.p'):
            # if it contains batch 1 remove it
                if 'batch1.' not in file_name:
                    if 'batch1_' not in file_name:    
                        normal_files.append(file_name)

    normal_files.sort()
    if len(normal_files) < 31:
        print(len(normal_files))
        print('!!! TO FEW DATA FILES !!!')
    
    if len(normal_files) > 31:
        normal_files = normal_files[:31]

    # add the data path to the file names
    normal_files = [data_path + file_name for file_name in normal_files]
    iii = 0
    for j, file_temp in enumerate(normal_files):
        data = pickle.load(open(file_temp, 'rb'))
        for i, sub_batch in enumerate(data):
            id_min = np.argmin(np.array(sub_batch)[:,-1])
            temp_store_values[i] = np.array(sub_batch)[id_min]
            iii += 1
            print(j*7 + i, iii, len(sub_batch), len(normal_files))
            try: number_of_points[j*8 + i] = len(sub_batch)
            except: print('hello',j*8 + i, BASE_NAME, DIM, RES, len(sub_batch), sub_batch, data.shape, len(normal_files))
        
        if dim == 4:
            big_array[j*8:(j+1)*8] = temp_store_values[:8]
        else:
            big_array[j*7:(j+1)*7] = temp_store_values[:7]


    print('-'*20)
    print(big_array)

    if save:
        pickle.dump(big_array, open('compressed/'+BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES) + '_combined.p', 'wb'))
        pickle.dump(number_of_points, open('compressed/'+BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES) + '_combined_points.p', 'wb'))
    return big_array, number_of_points

# find all files starting with true_hyper_d_3_res_5 in the file names list
def combine_data_rank_0_grid_rand(BASE_NAME, DIM,RES, save = False):
    big_array = np.zeros((8*31, DIM+1))
    temp_store_values = np.zeros((8, DIM+1))
    normal_files = []
    for file_name in file_names:
        if file_name.startswith(BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES)):
            if not file_name.endswith('_rank_0.p'):
            # if it contains batch 1 remove it
                if 'batch1.' not in file_name:
                    if 'batch1_' not in file_name:    
                        normal_files.append(file_name)
    normal_files.sort()
    if len(normal_files) < 31:
        print(len(normal_files))
        print('!!! TO FEW DATA FILES !!!')
    # add the data path to the file names
    normal_files = [data_path + file_name for file_name in normal_files]
    for j, file_temp in enumerate(normal_files):
        data = pickle.load(open(file_temp, 'rb'))
        data_rank_0 = np.array(pickle.load(open(file_temp[:-2] + '_rank_0.p', 'rb')))[0]
        temp_store_values[0] = data_rank_0[np.argmin(data_rank_0[:,-1])]
        for i, sub_batch in enumerate(data):
            id_min = np.argmin(np.array(sub_batch)[:,-1])
            temp_store_values[i+1] = np.array(sub_batch)[id_min]
        big_array[j*8:(j+1)*8] = temp_store_values
        if dim == 4:
            big_array[j*8:(j+1)*8] = temp_store_values[:8]
        else:
            big_array[j*7:(j+1)*7] = temp_store_values[:7]

    if save:
        pickle.dump(big_array, open('compressed/'+BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES) + '_combined.p', 'wb'))
    return big_array

# find all files starting with true_hyper_d_3_res_5 in the file names list
def combine_data(BASE_NAME, DIM,RES, save = False):
    big_array = np.zeros((8*31, DIM+1))
    temp_store_values = np.zeros((8, DIM+1))
    normal_files = []
    for file_name in file_names:
        if file_name.startswith(BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES)):
            if file_name[14:19] == 'batch':
                normal_files.append(file_name)
    normal_files.sort()
    if len(normal_files) < 31:
        print(len(normal_files))
        print('!!! TO FEW DATA FILES !!!')
    if len(normal_files) > 31: 
        normal_files = normal_files[:31]
    # add the data path to the file names
    normal_files = [data_path + file_name for file_name in normal_files]
    for j, file_temp in enumerate(normal_files):
        data = pickle.load(open(file_temp, 'rb'))
        for i, sub_batch in enumerate(data):
            id_min = np.argmin(np.array(sub_batch)[:,-1])
            temp_store_values[i] = np.array(sub_batch)[id_min]
        if j*8 == 248:
            break
        if dim == 4:
            big_array[j*8:(j+1)*8] = temp_store_values[:8]
        else:
            big_array[j*7:(j+1)*7] = temp_store_values[:7]

    if save:
        pickle.dump(big_array, open('compressed/'+BASE_NAME + '_d_' + str(DIM) + '_res_' + str(RES) + '_combined.p', 'wb'))
    return big_array


base = base_name[0]
for dim in dimension:
    for res in resolution:
        big_out, points_out = combine_data_rank_0(base, dim, res, save = True)
        print(base, dim, res)
        print(big_out.shape)
        print('------------------')
    
base = base_name[1]
for dim in dimension:
    for res in resolution:

        big_out, points_out = combine_data_rank_0(base, dim, res, save = True)
        if res == 5:
            print(big_out)
            print(np.sum(big_out == 0))
        print(base, dim, res)
        print(big_out.shape)
        print('------------------')

for base in base_name[2:]:
    for dim in dimension:
        for res in resolution:
            big_out = combine_data(base, dim, res, save = True)
            print(base, dim, res)
            print(big_out.shape)
            print('------------------')