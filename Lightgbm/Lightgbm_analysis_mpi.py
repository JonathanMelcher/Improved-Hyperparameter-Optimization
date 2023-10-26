"""
                Module for the Lightgbm analysis run with MPI


This module tests different grid types for the Lightgbm algorithm on
batches of the KDD12 dataset. The grid types tested are: regular grid, random
uniform grid, latin hypercube sampling, random sampling and RSA. The module
is run with MPI4py build against OpenMPI.

"""


import pickle
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from lightgbm import early_stopping
import Make_grids as mg


# Pipeline to change the resolution and dimension of the grid
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--res', type=int, default=3,
                    help='resolution of grid', )
parser.add_argument('--dim', type=int, default=2,
                    help='dimension of grid', )

resolution = np.array([parser.parse_args().res]*parser.parse_args().dim)

# predifened domain for the grid
edges = np.array([[0.001,3,5,0.5], [1,20,50,0.99]])

edges = edges[:,:parser.parse_args().dim]

# MPI setup make sure it is build against OpenMPI
from mpi4py import MPI
import socket

if not MPI.Is_initialized():
    MPI.Init()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(MPI.Get_library_version())

comm = MPI.COMM_WORLD
print("Rank: ", comm.Get_rank(),'out of', size, "on Host: ", socket.gethostname())

if rank == 0:
    print('res', resolution)
    print('edges', edges)


#NOTE change this for your own path
DATA_PATH = '../../../data/LGBM/Data/Batches/Batches_proto_4_csv/'

def run_iteration(i, X_train, X_val, X_test, y_train, y_val, y_test):
    Estimators = 750
    grid_size_check = len(i)
    Objective = 'l2_root'
    if grid_size_check == 2:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, objective=Objective)
        LGBM.fit(X_train, y_train, eval_set=[(X_val, y_val)],callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 3:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), objective=Objective)
        LGBM.fit(X_train, y_train, eval_set=[(X_val, y_val)],callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        batch_nr = [*i, mse]
        return batch_nr
    elif grid_size_check == 4:
        LGBM = LGBMRegressor(learning_rate=i[0], random_state=42, max_depth=int(i[1]), n_estimators = Estimators, num_leaves=int(i[2]), bagging_fraction=i[3], objective=Objective)
        LGBM.fit(X_train, y_train, eval_set=[(X_val, y_val)],callbacks = [early_stopping(10, first_metric_only=True, verbose=False)])
        y_pred = LGBM.predict(X_test)
        mse = np.sqrt(mean_squared_error(y_test, y_pred))
        i[1] = int(i[1])
        i[2] = int(i[2])
        batch_nr = [*i, mse]
        return batch_nr




# define a function that takes a grid and returns the median minimum loss for each batch
def median_min_loss(grid):
    """
    Function that takes a grid and returns the median minimum loss for each 
    batch and controls some of the MPI communication between the workers 
    and the master node.

    Parameters
    ----------
    grid : array
        The grid to be tested. From the Make_grids module.
    
    Returns
    -------
    results_par_save : array
        The median minimum loss for each batch.
    """
    if rank == 0:
        print('master setup for reciving')
        # setup structure for saving results
        results_par_save = []
        # setup structure for receiving results from all workers

        for i in range(1,size):
            results_par_save.append(comm.recv(source=i, tag=11))
        
        print('master recived all packages')
        results_par_save = np.array(results_par_save, dtype=object)
        

        return results_par_save
    
    results_par_save_worker = []

    data = pd.read_csv(DATA_PATH + 'kdd12_batch_{}.csv'.format(rank))
    X_train = data.drop('2',axis=1)
    y_train = data['2']
    # split 70/30 train/test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=1999)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    for point in grid:
        results = run_iteration(point, X_train, X_val, X_test, y_train, y_val, y_test)
        results_par_save_worker.append(results)
        
    # send results to master
    comm.send(results_par_save_worker, dest=0, tag=11)
    
    return None, None




def totalt_analysis(resolution, edges, t, k):
    d = len(resolution)

    # RSA
    if rank == 0:
        print(f'{rank} started RSA')
    values_rsa = median_min_loss(mg.rsa_grid(resolution, edges))

    if rank == 0:
        pickle.dump(values_rsa, open('../../../data/GeneratedData/LGBM/rsa_d_{}_res_{}.p'.format(d, resolution[0]), 'wb'))

    #regular grid
    if rank == 0:
        print(f'{rank} started regular grid')
    values_reg = median_min_loss(mg.reg_grid(resolution, edges))
    if rank == 0:
        pickle.dump(values_reg, open('../../../data/GeneratedData/LGBM/reg_d_{}_res_{}.p'.format(d, resolution[0]), 'wb'))


    #random uniform grid
    if rank == 0:
        print(f'{rank} started random uniform grid')
    values_rand = median_min_loss(mg.random_grid(resolution, edges))

    if rank == 0:
        pickle.dump(values_rand, open('../../../data/GeneratedData/LGBM/ran_d_{}_res_{}.p'.format(d, resolution[0]), 'wb'))


    #hypertilling
    if rank == 0:
        print(f'{rank} started hypertilling')
    values_hyper = median_min_loss(mg.hyper_tilling_grid(resolution, edges, t, k))
    
    if rank == 0:
       pickle.dump(values_hyper, open('../../../data/GeneratedData/LGBM/hyp_d_{}_res_{}.p'.format(d, resolution[0]), 'wb'))
    

    #latin hypercube sampling
    if rank == 0:
        print(f'{rank} started LHS')
    values_lhs = median_min_loss(mg.lhs_grid(resolution, edges))

    if rank == 0:
       pickle.dump(values_lhs, open('../../../data/GeneratedData/LGBM/lhs_d_{}_res_{}.p'.format(d, resolution[0]), 'wb'))

    if rank == 0:

        return True
    
    else:
        return None




if rank == 0:
    tot_out =totalt_analysis(resolution, edges, 6, 1)
    if tot_out == True:
        print('Successfull run')
else:
    totalt_analysis(resolution, edges, 6, 1)